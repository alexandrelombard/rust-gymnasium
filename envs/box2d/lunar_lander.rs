use crate::core::{Env, Info, RenderFrame, Step};
use crate::utils::render2d::{Canvas, BLACK, BLUE, GRAY, GREEN, RED, WHITE};
use crate::utils::rng::{rng_from_seed, RngStream};
use rand::distributions::Distribution;

/// A lightweight, dependency-free approximation of Gymnasium's LunarLander-v2
/// environment. Physics are simplified but the interface matches:
/// - Observation: [x, y, vx, vy, angle, angular_velocity, left_contact, right_contact]
/// - Action space: Discrete(4) {0: do nothing, 1: left engine, 2: main engine, 3: right engine}
/// - Reward: shaped loosely to encourage soft upright landing on the pad.
/// - Episode terminates on land (success) or crash/out-of-bounds.
pub struct LunarLanderEnv {
    // State in world units (meters-ish, abstract)
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    angle: f32,
    vang: f32,
    left_contact: bool,
    right_contact: bool,

    steps: u32,
    pub max_episode_steps: u32,

    // Simulation parameters
    gravity: f32,      // m/s^2 downward
    main_thrust: f32,  // upward thrust accel when firing main
    side_thrust: f32,  // lateral+rotational accel for side engines
    ang_damp: f32,     // angular damping
    lin_damp: f32,     // linear damping
    dt: f32,           // time step

    // Bounds
    x_limit: f32,
    y_limit: f32,

    // Landing pad window (x in [-pad_w, pad_w] at y=0 is OK)
    pad_half_width: f32,

    rng: RngStream,
}

impl Default for LunarLanderEnv { fn default() -> Self { Self::new(2025) } }

impl LunarLanderEnv {
    pub fn new(seed: u64) -> Self {
        Self {
            x: 0.0,
            y: 1.0,
            vx: 0.0,
            vy: 0.0,
            angle: 0.0,
            vang: 0.0,
            left_contact: false,
            right_contact: false,
            steps: 0,
            max_episode_steps: 1000,
            gravity: 0.6,
            main_thrust: 1.2,
            side_thrust: 0.6,
            ang_damp: 0.15,
            lin_damp: 0.02,
            dt: 0.05,
            x_limit: 1.2,
            y_limit: 1.5,
            pad_half_width: 0.2,
            rng: rng_from_seed(seed),
        }
    }

    fn obs(&self) -> [f32; 8] {
        [
            self.x,
            self.y,
            self.vx,
            self.vy,
            self.angle,
            self.vang,
            if self.left_contact { 1.0 } else { 0.0 },
            if self.right_contact { 1.0 } else { 0.0 },
        ]
    }

    fn landed_success(&self) -> bool {
        // Landed if y <= 0 with small speeds and near upright and within pad x-range
        self.y <= 0.0
            && self.vy.abs() < 0.3
            && self.vx.abs() < 0.3
            && self.angle.abs() < 0.2
            && self.x.abs() <= self.pad_half_width
    }

    fn crashed(&self) -> bool {
        // Crash if below ground with high speed/tilt or out of bounds
        (self.y <= 0.0 && (self.vy.abs() >= 0.6 || self.angle.abs() >= 0.4))
            || self.x.abs() > self.x_limit
            || self.y > self.y_limit
            || self.y < -0.2
    }

    fn apply_action(&mut self, action: u32) {
        // Convert action into forces/torques. Angle 0 is upright; +angle rotates clockwise.
        let dt = self.dt;
        // Main engine: thrust along body up (world up adjusted by angle)
        if action == 2 {
            // thrust direction: opposite gravity in body frame rotated by angle
            let ca = self.angle.cos();
            let sa = self.angle.sin();
            let ax = -sa * self.main_thrust;
            let ay = ca * self.main_thrust;
            self.vx += ax * dt;
            self.vy += ay * dt;
        }
        // Left engine fires on left side to push right and rotate left (negative angle rate)
        if action == 1 {
            self.vx += self.side_thrust * dt;
            self.vang -= 1.5 * self.side_thrust * dt;
        }
        // Right engine fires to push left and rotate right (positive angle rate)
        if action == 3 {
            self.vx -= self.side_thrust * dt;
            self.vang += 1.5 * self.side_thrust * dt;
        }
    }

    pub fn render_pixels(&self, width: u32, height: u32) -> RenderFrame {
        let mut canvas = Canvas::new(width.max(400), height.max(300));
        let w = canvas.width as i32; let h = canvas.height as i32;
        canvas.clear(BLACK);

        // Ground line at y=0 world -> convert to screen
        let margin = 20;
        let world_h = self.y_limit + 0.2; // from -0.2 to y_limit
        let world_w = self.x_limit * 2.0;
        let to_sx = |x: f32| -> i32 {
            let t = ((x + self.x_limit) / world_w).clamp(0.0, 1.0);
            margin + (t * (w - 2 * margin) as f32) as i32
        };
        let to_sy = |y: f32| -> i32 {
            let t = ((y + 0.2) / world_h).clamp(0.0, 1.0);
            // screen y grows down, world y grows up
            margin + ((1.0 - t) * (h - 2 * margin) as f32) as i32
        };

        // Draw ground
        let gy = to_sy(0.0);
        canvas.draw_line(margin, gy, w - margin, gy, GREEN);
        // Draw pad
        let px0 = to_sx(-self.pad_half_width);
        let px1 = to_sx(self.pad_half_width);
        canvas.draw_line(px0, gy, px1, gy, WHITE);

        // Draw lander body as a rotated rectangle centered at (x,y)
        let cx = to_sx(self.x); let cy = to_sy(self.y);
        let body_w = 18; let body_h = 24;
        // approximate as axis-aligned rect for simplicity (ignore rotation for drawing body)
        canvas.fill_rect(cx - body_w / 2, cy - body_h / 2, body_w, body_h, BLUE);

        // Draw legs and contacts
        let leg_h = 10;
        canvas.draw_line(cx - body_w / 2, cy + body_h / 2, cx - body_w / 2 - 6, cy + body_h / 2 + leg_h, GRAY);
        canvas.draw_line(cx + body_w / 2, cy + body_h / 2, cx + body_w / 2 + 6, cy + body_h / 2 + leg_h, GRAY);
        if self.left_contact {
            canvas.fill_rect(cx - body_w / 2 - 7, cy + body_h / 2 + leg_h - 2, 4, 4, GREEN);
        }
        if self.right_contact {
            canvas.fill_rect(cx + body_w / 2 + 3, cy + body_h / 2 + leg_h - 2, 4, 4, GREEN);
        }

        // Draw a simple orientation line to visualize angle
        let line_len = 14.0;
        let ex = cx as f32 + line_len * self.angle.sin();
        let ey = cy as f32 - line_len * self.angle.cos();
        canvas.draw_line(cx, cy, ex as i32, ey as i32, WHITE);

        canvas.into_render_frame()
    }
}

impl Env for LunarLanderEnv {
    type Obs = [f32; 8];
    type Act = u32; // 0..=3

    fn reset(&mut self, seed: Option<u64>) -> (Self::Obs, Info) {
        if let Some(s) = seed { self.rng = rng_from_seed(s); }
        use rand::distributions::Uniform;
        let uni_x = Uniform::new_inclusive(-0.05f32, 0.05f32);
        let uni_a = Uniform::new_inclusive(-0.1f32, 0.1f32);
        self.x = uni_x.sample(&mut self.rng);
        self.y = 1.2; // start above ground
        self.vx = 0.0;
        self.vy = 0.0;
        self.angle = uni_a.sample(&mut self.rng);
        self.vang = 0.0;
        self.left_contact = false;
        self.right_contact = false;
        self.steps = 0;
        (self.obs(), Info::new())
    }

    fn step(&mut self, action: Self::Act) -> Step<Self::Obs> {
        // physics integration
        let dt = self.dt;
        // Apply gravity
        self.vy -= self.gravity * dt;
        // Apply action effects
        self.apply_action(action.min(3));

        // Damping
        self.vx *= 1.0 - self.lin_damp;
        self.vy *= 1.0 - self.lin_damp * 0.5;
        self.vang *= 1.0 - self.ang_damp;

        // Integrate
        self.x += self.vx * dt;
        self.y += self.vy * dt;
        self.angle += self.vang * dt;
        // keep angle in [-pi, pi]
        let pi = std::f32::consts::PI; let two_pi = 2.0 * pi;
        self.angle = ((self.angle + pi) % two_pi + two_pi) % two_pi - pi;

        // Ground collision and contacts
        self.left_contact = false; self.right_contact = false;
        if self.y <= 0.0 {
            // Detect a basic two-point contact from body x offset
            let left_on_pad = (self.x - 0.05).abs() <= self.pad_half_width;
            let right_on_pad = (self.x + 0.05).abs() <= self.pad_half_width;
            self.left_contact = left_on_pad;
            self.right_contact = right_on_pad;
            // Clamp at ground
            self.y = 0.0;
            self.vy = 0.0;
        }

        self.steps += 1;
        let terminated = self.landed_success() || self.crashed();
        let truncated = self.steps >= self.max_episode_steps;

        // Reward shaping: distance to pad center, penalty for tilt and speed; big reward for success
        let mut reward = 0.0;
        reward -= (self.x / self.x_limit).abs() * 0.5;
        reward -= (self.vx.abs() + self.vy.abs()) * 0.1;
        reward -= self.angle.abs() * 0.2;
        if self.landed_success() { reward += 100.0; }
        if self.crashed() { reward -= 100.0; }

        Step::new(self.obs(), reward, terminated, truncated, Info::new())
    }

    fn render(&self) -> Option<RenderFrame> {
        Some(self.render_pixels(400, 300))
    }

    fn close(&mut self) {}
}
