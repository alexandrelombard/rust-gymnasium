use rust_gymnasium::{AcrobotEnv, CartPoleEnv, Env, MountainCarEnv, PendulumEnv, RenderFrame};
use minifb::{Key, Window, WindowOptions};
use rand::Rng;

fn rgba_to_u32(a: u8, r: u8, g: u8, b: u8) -> u32 {
    // Minifb expects ARGB on most platforms; construct accordingly.
    ((a as u32) << 24) | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

fn main() {
    let mut env = CartPoleEnv::default();
    let (_obs, _info) = env.reset(Some(123));

    // Initial render to get frame size
    let mut frame = env.render().expect("render() should produce a frame");
    let (width, height) = match &frame {
        RenderFrame::Pixels { width, height, .. } => (*width as usize, *height as usize),
        RenderFrame::Text(_) => {
            // If environment renders text by default, fall back to a pixel render helper if available
            // For CartPoleEnv, render() returns pixels. For other envs, just exit gracefully.
            eprintln!("Environment returned text frame; nothing to show in a pixel window.");
            return;
        }
    };

    let mut window = Window::new(
        "rust-gymnasium: run_rand_render",
        width,
        height,
        WindowOptions::default(),
    ).expect("Unable to open window");

    let mut buffer: Vec<u32> = vec![0; width * height];
    let mut rng = rand::thread_rng();

    // Run until window is closed or Escape pressed
    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Take a random action each frame
        let action: u32 = if rng.r#gen::<f32>() < 0.5 { 0 } else { 1 };
        let step = env.step(action);
        if step.terminated || step.truncated {
            let _ = env.reset(None);
        }

        // Render current frame
        frame = env.render().expect("render() should produce a frame");
        if let RenderFrame::Pixels { width, height, data } = frame {
            // Convert RGBA bytes to ARGB u32s for minifb
            for i in 0..(width as usize * height as usize) {
                let idx = i * 4;
                let r = data[idx];
                let g = data[idx + 1];
                let b = data[idx + 2];
                let a = data[idx + 3];
                buffer[i] = rgba_to_u32(a, r, g, b);
            }
            window.update_with_buffer(&buffer, width as usize, height as usize)
                .expect("Failed to update window buffer");
        } else {
            // If render switched to text unexpectedly, just break
            break;
        }
    }
}
