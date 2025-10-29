use crate::core::{Env, Info, RenderFrame, Step};
use crate::utils::rng::{rng_from_seed, RngStream};
use crate::utils::render2d::{Canvas, BLUE, GRAY, GREEN, RED, WHITE};
use rand::distributions::Distribution;

/// MountainCarContinuous-v0 environment (Gymnasium classic_control)
/// Observation: [position, velocity]
/// Action space: Box([-1.0], [1.0]) -> scalar throttle in [-1, 1]
/// Reward: -0.1 * action^2 each step; +100 bonus and terminate when reaching the goal.
/// Episode truncates at max steps (default 999)
pub struct MountainCarContinuousEnv {
    position: f32,
    velocity: f32,

    pub max_episode_steps: u32,
    steps: u32,

    rng: RngStream,

    // Constants (matching Gymnasium defaults)
    min_position: f32, // -1.2
    max_position: f32, // 0.6
    max_speed: f32,    // 0.07
    goal_position: f32, // 0.45 in continuous variant
    power: f32,        // 0.0015 (action scaled by this)
    gravity: f32,      // 0.0025
}

impl Default for MountainCarContinuousEnv { fn default() -> Self { Self::new(2024) } }

impl MountainCarContinuousEnv {
    pub fn new(seed: u64) -> Self {
        Self {
            position: 0.0,
            velocity: 0.0,
            max_episode_steps: 999,
            steps: 0,
            rng: rng_from_seed(seed),
            min_position: -1.2,
            max_position: 0.6,
            max_speed: 0.07,
            goal_position: 0.45,
            power: 0.0015,
            gravity: 0.0025,
        }
    }

    fn obs(&self) -> [f32; 2] { [self.position, self.velocity] }

    /// 2D pixel rendering similar to MountainCarEnv (discrete)
    pub fn render_pixels(&self, width: u32, height: u32) -> RenderFrame {
        let mut canvas = Canvas::new(width.max(320), height.max(240));
        canvas.clear(WHITE);

        let w = canvas.width as i32;
        let h = canvas.height as i32;
        let margin = 20;

        // Terrain parameters
        let base_y = (h as f32 * 0.7) as i32; // baseline of terrain
        let amp = (h as f32 * 0.25) as i32;   // amplitude of hills

        // World to screen mapping for x in [min_position, max_position]
        let world_min = self.min_position;
        let world_max = self.max_position;
        let usable = (w - 2 * margin) as f32;
        let to_screen_x = |xw: f32| -> i32 {
            let t = ((xw - world_min) / (world_max - world_min)).clamp(0.0, 1.0);
            margin as i32 + (t * usable) as i32
        };
        let terrain_y = |xw: f32| -> i32 {
            let y = (3.0 * xw).sin(); // ~[-1, 1]
            base_y - ((y * amp as f32) as i32)
        };

        // Draw terrain as a polyline
        let steps = w.max(2);
        let mut prev_x = 0;
        let mut prev_y = 0;
        for i in 0..steps {
            let t = i as f32 / (steps - 1) as f32;
            let xw = world_min + t * (world_max - world_min);
            let xs = to_screen_x(xw);
            let ys = terrain_y(xw);
            if i > 0 { canvas.draw_line(prev_x, prev_y, xs, ys, GREEN); }
            prev_x = xs;
            prev_y = ys;
        }

        // Draw goal flag at goal_position
        let goal_x = to_screen_x(self.goal_position);
        let goal_y = terrain_y(self.goal_position);
        let flag_h = 25;
        canvas.draw_line(goal_x, goal_y, goal_x, goal_y - flag_h, RED);
        canvas.draw_line(goal_x, goal_y - flag_h, goal_x + 10, goal_y - flag_h + 6, RED);

        // Draw the car as a small rectangle centered at current position sitting on terrain
        let car_xs = to_screen_x(self.position);
        let car_ys = terrain_y(self.position);
        let car_w = 20;
        let car_h = 12;
        canvas.fill_rect(car_xs - car_w / 2, car_ys - car_h - 2, car_w, car_h, BLUE);

        // Draw a simple velocity indicator above the car
        let vx = (self.velocity * 100.0).clamp(-30.0, 30.0) as i32; // scale for visibility
        canvas.draw_line(car_xs, car_ys - car_h - 6, car_xs + vx, car_ys - car_h - 6, GRAY);

        canvas.into_render_frame()
    }
}

impl Env for MountainCarContinuousEnv {
    type Obs = [f32; 2];
    type Act = f32; // continuous scalar in [-1, 1]

    fn reset(&mut self, seed: Option<u64>) -> (Self::Obs, Info) {
        if let Some(s) = seed { self.rng = rng_from_seed(s); }
        let mut uni = rand::distributions::Uniform::new_inclusive(-0.6f32, -0.4f32);
        self.position = uni.sample(&mut self.rng);
        self.velocity = 0.0;
        self.steps = 0;
        (self.obs(), Info::new())
    }

    fn step(&mut self, action: Self::Act) -> Step<Self::Obs> {
        // Clip action to [-1, 1]
        let a = action.clamp(-1.0, 1.0);
        // Dynamics as in Gymnasium MountainCarContinuous
        self.velocity += a * self.power - self.gravity * (3.0 * self.position).cos();
        if self.velocity > self.max_speed { self.velocity = self.max_speed; }
        if self.velocity < -self.max_speed { self.velocity = -self.max_speed; }
        self.position += self.velocity;
        if self.position > self.max_position { self.position = self.max_position; }
        if self.position < self.min_position { self.position = self.min_position; }
        if self.position <= self.min_position && self.velocity < 0.0 { self.velocity = 0.0; }

        self.steps += 1;
        let mut reward = -0.1 * a * a; // action penalty each step
        let terminated = self.position >= self.goal_position;
        if terminated { reward += 100.0; }
        let truncated = self.steps >= self.max_episode_steps;
        Step::new(self.obs(), reward, terminated, truncated, Info::new())
    }

    fn render(&self) -> Option<RenderFrame> { Some(self.render_pixels(320, 240)) }

    fn close(&mut self) {}
}
