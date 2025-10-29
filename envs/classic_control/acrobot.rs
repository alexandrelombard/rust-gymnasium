use crate::core::{Env, Info, RenderFrame, Step};
use crate::utils::rng::{rng_from_seed, RngStream};
use crate::utils::render2d::{Canvas, Color, BLACK, BLUE, GRAY, GREEN, RED, WHITE};
use rand::distributions::Distribution;

/// Acrobot-v1 environment (Gymnasium classic_control)
/// Underactuated double pendulum with actuation on the second joint.
/// Action space: Discrete(3) torques {-1.0, 0.0, +1.0}
/// Observation (simplified here): [theta1, theta2, dtheta1, dtheta2]
/// Reward: -1.0 per step until the terminal height is reached; truncates at max steps.
pub struct AcrobotEnv {
    // State
    th1: f32,
    th2: f32,
    dth1: f32,
    dth2: f32,

    steps: u32,
    pub max_episode_steps: u32,

    rng: RngStream,

    // Constants (following Gymnasium)
    m1: f32,
    m2: f32,
    l1: f32,
    l2: f32,
    lc1: f32,
    lc2: f32,
    I1: f32,
    I2: f32,
    g: f32,
    dt: f32, // integration timestep

    // Limits
    max_vel_1: f32,
    max_vel_2: f32,
}

impl Default for AcrobotEnv {
    fn default() -> Self { Self::new(7) }
}

impl AcrobotEnv {
    pub fn new(seed: u64) -> Self {
        Self {
            th1: 0.0,
            th2: 0.0,
            dth1: 0.0,
            dth2: 0.0,
            steps: 0,
            max_episode_steps: 500,
            rng: rng_from_seed(seed),
            m1: 1.0,
            m2: 1.0,
            l1: 1.0,
            l2: 1.0,
            lc1: 0.5,
            lc2: 0.5,
            I1: 1.0,
            I2: 1.0,
            g: 9.8,
            dt: 0.2,
            max_vel_1: 4.0 * std::f32::consts::PI,
            max_vel_2: 9.0 * std::f32::consts::PI,
        }
    }

    fn obs(&self) -> [f32; 4] { [self.th1, self.th2, self.dth1, self.dth2] }

    fn terminal_height_reached(&self) -> bool {
        // Terminal when -cos(th1) - cos(th1+th2) >= 1.0
        let h = -self.th1.cos() - (self.th1 + self.th2).cos();
        h >= 1.0
    }

    pub fn render_pixels(&self, width: u32, height: u32) -> RenderFrame {
        let mut canvas = Canvas::new(width.max(400), height.max(400));
        canvas.clear(WHITE);
        let w = canvas.width as i32;
        let h = canvas.height as i32;

        // Pivot at top center with margin
        let margin = 40;
        let px = w / 2;
        let py = margin;

        // Scale: total length l1+l2 maps to 80% of available height below pivot
        let total_len = self.l1 + self.l2; // in meters (units)
        let avail = (h - py - margin) as f32;
        let scale = 0.8f32 * avail / total_len.max(1e-6);

        // Compute joint positions: angles measured from vertical, positive CCW
        let th1 = self.th1;
        let th2 = self.th2;
        let x1 = px as f32 + self.l1 * th1.sin() * scale;
        let y1 = py as f32 + self.l1 * th1.cos() * scale;
        let th12 = th1 + th2;
        let x2 = x1 + self.l2 * th12.sin() * scale;
        let y2 = y1 + self.l2 * th12.cos() * scale;

        // Base mount
        canvas.fill_rect(px - 20, py - 8, 40, 8, GRAY);
        // First link (draw a slightly thicker line by multiple strokes)
        canvas.draw_line(px, py, x1 as i32, y1 as i32, BLUE);
        canvas.draw_line(px + 1, py, x1 as i32 + 1, y1 as i32, BLUE);
        canvas.draw_line(px, py + 1, x1 as i32, y1 as i32 + 1, BLUE);
        // Second link
        canvas.draw_line(x1 as i32, y1 as i32, x2 as i32, y2 as i32, GREEN);
        canvas.draw_line(x1 as i32 + 1, y1 as i32, x2 as i32 + 1, y2 as i32, GREEN);
        canvas.draw_line(x1 as i32, y1 as i32 + 1, x2 as i32, y2 as i32 + 1, GREEN);

        // Joints as small squares
        let r = 4;
        canvas.fill_rect(px - r, py - r, 2 * r, 2 * r, RED);
        canvas.fill_rect(x1 as i32 - r, y1 as i32 - r, 2 * r, 2 * r, RED);
        canvas.fill_rect(x2 as i32 - r, y2 as i32 - r, 2 * r, 2 * r, RED);

        // Optional: step counter
        // canvas.fill_rect(10, 10, 0, 0, BLACK); // placeholder if text not supported

        canvas.into_render_frame()
    }

    fn dynamics(&self, th1: f32, th2: f32, dth1: f32, dth2: f32, torque: f32) -> (f32, f32) {
        // Equations adapted from Gymnasium's acrobot implementation
        let m1 = self.m1; let m2 = self.m2;
        let l1 = self.l1; let l2 = self.l2;
        let lc1 = self.lc1; let lc2 = self.lc2;
        let I1 = self.I1; let I2 = self.I2;
        let g = self.g;
        let d1 = m1 * lc1 * lc1 + m2 * (l1 * l1 + lc2 * lc2 + 2.0 * l1 * lc2 * (th2).cos()) + I1 + I2;
        let d2 = m2 * (lc2 * lc2 + l1 * lc2 * (th2).cos()) + I2;
        let phi2 = m2 * lc2 * g * (th1 + th2 - std::f32::consts::FRAC_PI_2).cos();
        let phi1 = -m2 * l1 * lc2 * dth2 * dth2 * (th2).sin()
            - 2.0 * m2 * l1 * lc2 * dth2 * dth1 * (th2).sin()
            + (m1 * lc1 + m2 * l1) * g * (th1 - std::f32::consts::FRAC_PI_2).cos()
            + phi2;
        let ddth2 = (torque + d2 / d1 * phi1 - m2 * l1 * lc2 * dth1 * dth1 * (th2).sin() - phi2)
            / (m2 * lc2 * lc2 + I2 - d2 * d2 / d1);
        let ddth1 = -(d2 * ddth2 + phi1) / d1;
        (ddth1, ddth2)
    }
}

impl Env for AcrobotEnv {
    type Obs = [f32; 4];
    type Act = u32; // 0,1,2

    fn reset(&mut self, seed: Option<u64>) -> (Self::Obs, Info) {
        if let Some(s) = seed { self.rng = rng_from_seed(s); }
        // Initialize th1, th2 uniformly in [-pi, pi], velocities in [-0.1, 0.1]
        use rand::distributions::Uniform;
        let mut u_ang = Uniform::new_inclusive(-std::f32::consts::PI, std::f32::consts::PI);
        let mut u_vel = Uniform::new_inclusive(-0.1f32, 0.1f32);
        self.th1 = u_ang.sample(&mut self.rng);
        self.th2 = u_ang.sample(&mut self.rng);
        self.dth1 = u_vel.sample(&mut self.rng);
        self.dth2 = u_vel.sample(&mut self.rng);
        self.steps = 0;
        (self.obs(), Info::new())
    }

    fn step(&mut self, action: Self::Act) -> Step<Self::Obs> {
        let torque = match action { 0 => -1.0, 1 => 0.0, _ => 1.0 };

        // Integrate using RK4 for better stability
        let dt = self.dt;
        let (k1_1, k1_2) = (self.dth1, self.dth2);
        let (a1, a2) = self.dynamics(self.th1, self.th2, self.dth1, self.dth2, torque);

        let th1_2 = self.th1 + 0.5 * dt * k1_1;
        let th2_2 = self.th2 + 0.5 * dt * k1_2;
        let dth1_2 = self.dth1 + 0.5 * dt * a1;
        let dth2_2 = self.dth2 + 0.5 * dt * a2;
        let (a1_2, a2_2) = self.dynamics(th1_2, th2_2, dth1_2, dth2_2, torque);

        let th1_3 = self.th1 + 0.5 * dt * dth1_2;
        let th2_3 = self.th2 + 0.5 * dt * dth2_2;
        let dth1_3 = self.dth1 + 0.5 * dt * a1_2;
        let dth2_3 = self.dth2 + 0.5 * dt * a2_2;
        let (a1_3, a2_3) = self.dynamics(th1_3, th2_3, dth1_3, dth2_3, torque);

        let th1_4 = self.th1 + dt * dth1_3;
        let th2_4 = self.th2 + dt * dth2_3;
        let dth1_4 = self.dth1 + dt * a1_3;
        let dth2_4 = self.dth2 + dt * a2_3;
        let (a1_4, a2_4) = self.dynamics(th1_4, th2_4, dth1_4, dth2_4, torque);

        self.th1 += dt * (k1_1 + 2.0 * dth1_2 + 2.0 * dth1_3 + dth1_4) / 6.0;
        self.th2 += dt * (k1_2 + 2.0 * dth2_2 + 2.0 * dth2_3 + dth2_4) / 6.0;
        self.dth1 += dt * (a1 + 2.0 * a1_2 + 2.0 * a1_3 + a1_4) / 6.0;
        self.dth2 += dt * (a2 + 2.0 * a2_2 + 2.0 * a2_3 + a2_4) / 6.0;

        // Wrap angles to [-pi, pi]
        let pi = std::f32::consts::PI;
        let two_pi = 2.0 * pi;
        self.th1 = ((self.th1 + pi) % two_pi + two_pi) % two_pi - pi;
        self.th2 = ((self.th2 + pi) % two_pi + two_pi) % two_pi - pi;

        // Clamp velocities
        if self.dth1 > self.max_vel_1 { self.dth1 = self.max_vel_1; }
        if self.dth1 < -self.max_vel_1 { self.dth1 = -self.max_vel_1; }
        if self.dth2 > self.max_vel_2 { self.dth2 = self.max_vel_2; }
        if self.dth2 < -self.max_vel_2 { self.dth2 = -self.max_vel_2; }

        self.steps += 1;
        let terminated = self.terminal_height_reached();
        let truncated = self.steps >= self.max_episode_steps;
        let reward = if terminated { 0.0 } else { -1.0 };
        Step::new(self.obs(), reward, terminated, truncated, Info::new())
    }

    fn render(&self) -> Option<RenderFrame> {
        Some(self.render_pixels(400, 400))
    }

    fn close(&mut self) {}
}
