use crate::core::{Env, Info, RenderFrame, Step};
use crate::utils::rng::{rng_from_seed, RngStream};
use crate::utils::render2d::{Canvas, BLACK, BLUE, GRAY, RED, WHITE};
use rand::distributions::Distribution;

/// Pendulum-v1 (simplified) classic control environment
/// State: angle (theta, radians), angular velocity (theta_dot)
/// Action: Discrete(3) -> torque in {-max_torque, 0, +max_torque}
/// Reward: -(theta_norm^2 + 0.1 * theta_dot^2 + 0.001 * torque^2)
/// Episode: no natural termination; we truncate at max_episode_steps
pub struct PendulumEnv {
    theta: f32,
    theta_dot: f32,

    pub max_episode_steps: u32,
    steps: u32,

    rng: RngStream,

    // constants
    g: f32,
    m: f32,
    l: f32,
    max_speed: f32,
    max_torque: f32,
    dt: f32,
}

impl Default for PendulumEnv { fn default() -> Self { Self::new(42) } }

impl PendulumEnv {
    pub fn new(seed: u64) -> Self {
        Self {
            theta: 0.0,
            theta_dot: 0.0,
            max_episode_steps: 200,
            steps: 0,
            rng: rng_from_seed(seed),
            g: 10.0,
            m: 1.0,
            l: 1.0,
            max_speed: 8.0,
            max_torque: 2.0,
            dt: 0.05,
        }
    }

    fn obs(&self) -> [f32; 3] {
        [self.theta.cos(), self.theta.sin(), self.theta_dot]
    }

    #[inline]
    fn angle_normalize(x: f32) -> f32 {
        let pi = std::f32::consts::PI;
        ((x + pi) % (2.0 * pi) + (2.0 * pi)) % (2.0 * pi) - pi
    }

    /// Simple 2D rendering: draw a pivot and a rod with a bob.
    pub fn render_pixels(&self, width: u32, height: u32) -> RenderFrame {
        let mut canvas = Canvas::new(width.max(320), height.max(240));
        canvas.clear(WHITE);
        let w = canvas.width as i32;
        let h = canvas.height as i32;
        let cx = w / 2;
        let cy = (h as f32 * 0.3) as i32; // pivot near top
        // rod length in pixels
        let rod_len = ((h as f32) * 0.45) as i32;
        // compute bob position
        let x = (self.theta.sin() * rod_len as f32) as i32;
        let y = (self.theta.cos() * rod_len as f32) as i32;
        let bx = cx + x;
        let by = cy + y;

        // draw support
        canvas.fill_rect(cx - 30, cy - 6, 60, 12, GRAY);
        // draw rod
        canvas.draw_line(cx, cy, bx, by, BLACK);
        // draw bob
        let bob_r = 10;
        // approximate filled circle by square + cross for simplicity
        canvas.fill_rect(bx - bob_r, by - bob_r, bob_r * 2, bob_r * 2, BLUE);
        canvas.draw_line(bx - bob_r, by, bx + bob_r, by, WHITE);
        canvas.draw_line(bx, by - bob_r, bx, by + bob_r, WHITE);

        // draw a small velocity arc indicator at pivot
        let vel_len = (self.theta_dot * 5.0).clamp(-40.0, 40.0) as i32;
        canvas.draw_line(cx, cy, cx + vel_len, cy, RED);

        canvas.into_render_frame()
    }
}

impl Env for PendulumEnv {
    type Obs = [f32; 3];
    type Act = u32; // 0,1,2

    fn reset(&mut self, seed: Option<u64>) -> (Self::Obs, Info) {
        if let Some(s) = seed { self.rng = rng_from_seed(s); }
        // theta ~ U[-pi, pi), theta_dot ~ U[-1, 1]
        let mut u_theta = rand::distributions::Uniform::new_inclusive(-std::f32::consts::PI, std::f32::consts::PI);
        let mut u_vel = rand::distributions::Uniform::new_inclusive(-1.0f32, 1.0f32);
        self.theta = u_theta.sample(&mut self.rng);
        self.theta_dot = u_vel.sample(&mut self.rng);
        self.steps = 0;
        (self.obs(), Info::new())
    }

    fn step(&mut self, action: Self::Act) -> Step<Self::Obs> {
        let torque = match action { 0 => -self.max_torque, 1 => 0.0, _ => self.max_torque };
        let g = self.g; let m = self.m; let l = self.l; let dt = self.dt;

        // Dynamics derived from classic pendulum
        // theta_ddot = (3g/(2l)) * sin(theta) + (3/(m l^2)) * u
        let theta_ddot = (3.0 * g / (2.0 * l)) * self.theta.sin() + (3.0 / (m * l * l)) * torque;
        self.theta_dot += theta_ddot * dt;
        if self.theta_dot > self.max_speed { self.theta_dot = self.max_speed; }
        if self.theta_dot < -self.max_speed { self.theta_dot = -self.max_speed; }
        self.theta += self.theta_dot * dt;
        self.theta = Self::angle_normalize(self.theta);

        self.steps += 1;
        let theta_norm = Self::angle_normalize(self.theta);
        let cost = theta_norm * theta_norm + 0.1 * self.theta_dot * self.theta_dot + 0.001 * torque * torque;
        let reward = -cost;
        let terminated = false; // pendulum has no terminal condition in v1
        let truncated = self.steps >= self.max_episode_steps;
        Step::new(self.obs(), reward, terminated, truncated, Info::new())
    }

    fn render(&self) -> Option<RenderFrame> { Some(self.render_pixels(320, 240)) }

    fn close(&mut self) {}
}
