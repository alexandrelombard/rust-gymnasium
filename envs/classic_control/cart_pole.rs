use crate::core::{Env, Info, RenderFrame, Step};
use crate::utils::rng::{rng_from_seed, RngStream};
use crate::utils::render2d::{Canvas, GRAY, WHITE, BLUE, RED};
use rand::distributions::Distribution;

/// CartPole-v1 environment (minimal faithful implementation of Gymnasium classic_control)
/// Observation: [x, x_dot, theta, theta_dot]
/// Action space: Discrete(2) {0: push left, 1: push right}
/// Reward: 1.0 per step until termination/truncation
pub struct CartPoleEnv {
    // State
    x: f32,
    x_dot: f32,
    theta: f32,
    theta_dot: f32,

    // Episode management
    steps: u32,
    pub max_episode_steps: u32,

    // RNG
    rng: RngStream,

    // Physics constants (from Gymnasium)
    gravity: f32,        // 9.8
    masscart: f32,       // 1.0
    masspole: f32,       // 0.1
    total_mass: f32,     // masscart + masspole
    length: f32,         // actually half the pole's length (0.5)
    polemass_length: f32,// masspole * length
    force_mag: f32,      // 10.0
    tau: f32,            // seconds between state updates (0.02)

    // Termination thresholds
    theta_threshold_radians: f32, // 12 degrees
    x_threshold: f32,             // 2.4
}

impl Default for CartPoleEnv {
    fn default() -> Self { Self::new(1_234_567) }
}

impl CartPoleEnv {
    pub fn new(seed: u64) -> Self {
        let gravity = 9.8f32;
        let masscart = 1.0;
        let masspole = 0.1;
        let total_mass = masscart + masspole;
        let length = 0.5; // half pole
        let polemass_length = masspole * length;
        let force_mag = 10.0;
        let tau = 0.02;
        let theta_threshold_radians = 12.0_f32.to_radians();
        let x_threshold = 2.4;
        Self {
            x: 0.0,
            x_dot: 0.0,
            theta: 0.0,
            theta_dot: 0.0,
            steps: 0,
            max_episode_steps: 500,
            rng: rng_from_seed(seed),
            gravity,
            masscart,
            masspole,
            total_mass,
            length,
            polemass_length,
            force_mag,
            tau,
            theta_threshold_radians,
            x_threshold,
        }
    }

    /// Returns a 2D pixel rendering of the current state with the requested size.
    /// This does not open a window; it just produces an RGBA pixel buffer.
    pub fn render_pixels(&self, width: u32, height: u32) -> RenderFrame {
        let mut canvas = Canvas::new(width.max(64), height.max(48));
        // Background
        canvas.clear(WHITE);

        let w = canvas.width as i32;
        let h = canvas.height as i32;
        let margin = 20;
        let track_y = (h as f32 * 0.75) as i32;

        // Draw ground/track
        canvas.draw_line(0, track_y, w - 1, track_y, GRAY);

        // Map world x in [-x_threshold, x_threshold] to screen x in [margin, w-margin]
        let usable = (w - 2 * margin) as f32;
        let xr = (self.x + self.x_threshold) / (2.0 * self.x_threshold); // 0..1
        let cart_cx = margin as f32 + xr.clamp(0.0, 1.0) * usable;

        // Cart size
        let cart_w = 50;
        let cart_h = 30;
        let cart_x = (cart_cx as i32) - cart_w / 2;
        let cart_y = track_y - cart_h; // sit on track
        canvas.fill_rect(cart_x, cart_y, cart_w, cart_h, BLUE);

        // Pole from top-center of cart. Theta is angle from vertical (0 upright)
        let pole_len = (h as f32 * 0.35) as i32; // relative to canvas height
        let top_x = cart_cx as i32;
        let top_y = cart_y; // attach at cart top center
        let theta = self.theta as f32;
        let end_x = top_x + (theta.sin() * pole_len as f32) as i32;
        let end_y = top_y - (theta.cos() * pole_len as f32) as i32;
        canvas.draw_line(top_x, top_y, end_x, end_y, RED);

        canvas.into_render_frame()
    }

    fn terminated(&self) -> bool {
        self.x < -self.x_threshold
            || self.x > self.x_threshold
            || self.theta < -self.theta_threshold_radians
            || self.theta > self.theta_threshold_radians
    }

    fn obs(&self) -> [f32; 4] { [self.x, self.x_dot, self.theta, self.theta_dot] }
}

impl Env for CartPoleEnv {
    type Obs = [f32; 4];
    type Act = u32; // 0 or 1

    fn reset(&mut self, seed: Option<u64>) -> (Self::Obs, Info) {
        if let Some(s) = seed { self.rng = rng_from_seed(s); }
        // small uniform noise in [-0.05, 0.05]
        let mut uni = rand::distributions::Uniform::new_inclusive(-0.05f32, 0.05f32);
        self.x = uni.sample(&mut self.rng);
        self.x_dot = uni.sample(&mut self.rng);
        self.theta = uni.sample(&mut self.rng);
        self.theta_dot = uni.sample(&mut self.rng);
        self.steps = 0;
        (self.obs(), Info::new())
    }

    fn step(&mut self, action: Self::Act) -> Step<Self::Obs> {
        // Action -> force
        let force = if action == 1 { self.force_mag } else { -self.force_mag };
        let cos_theta = self.theta.cos();
        let sin_theta = self.theta.sin();

        // for the interested, same equations as Gymnasium
        let temp = (force + self.polemass_length * self.theta_dot.powi(2) * sin_theta) / self.total_mass;
        let theta_acc = (self.gravity * sin_theta - cos_theta * temp)
            / (self.length * (4.0 / 3.0 - self.masspole * cos_theta.powi(2) / self.total_mass));
        let x_acc = temp - self.polemass_length * theta_acc * cos_theta / self.total_mass;

        // Euler integration
        self.x += self.tau * self.x_dot;
        self.x_dot += self.tau * x_acc;
        self.theta += self.tau * self.theta_dot;
        self.theta_dot += self.tau * theta_acc;

        self.steps += 1;
        let terminated = self.terminated();
        let truncated = self.steps >= self.max_episode_steps;
        let reward = 1.0;
        Step::new(self.obs(), reward, terminated, truncated, Info::new())
    }

    fn render(&self) -> Option<RenderFrame> {
        // Return a 2D pixel rendering by default.
        // Choose a reasonable default canvas size.
        Some(self.render_pixels(320, 240))
    }

    fn close(&mut self) {}
}
