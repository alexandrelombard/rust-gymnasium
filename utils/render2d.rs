use crate::core::RenderFrame;

#[derive(Clone, Copy, Debug)]
pub struct Color(pub u8, pub u8, pub u8, pub u8);

pub const BLACK: Color = Color(0, 0, 0, 255);
pub const WHITE: Color = Color(255, 255, 255, 255);
pub const RED: Color = Color(220, 20, 60, 255);
pub const GREEN: Color = Color(0, 200, 0, 255);
pub const BLUE: Color = Color(0, 120, 255, 255);
pub const GRAY: Color = Color(180, 180, 180, 255);

/// A minimal RGBA software canvas for simple 2D rendering.
pub struct Canvas {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>, // RGBA
}

impl Canvas {
    pub fn new(width: u32, height: u32) -> Self {
        let mut canvas = Self {
            width,
            height,
            pixels: vec![0; (width as usize) * (height as usize) * 4],
        };
        canvas.clear(BLACK);
        canvas
    }

    pub fn clear(&mut self, color: Color) {
        for y in 0..self.height as usize {
            for x in 0..self.width as usize {
                let idx = (y * self.width as usize + x) * 4;
                self.pixels[idx] = color.0;
                self.pixels[idx + 1] = color.1;
                self.pixels[idx + 2] = color.2;
                self.pixels[idx + 3] = color.3;
            }
        }
    }

    #[inline]
    pub fn put_pixel(&mut self, x: i32, y: i32, color: Color) {
        if x < 0 || y < 0 { return; }
        let (x, y) = (x as u32, y as u32);
        if x >= self.width || y >= self.height { return; }
        let idx = ((y * self.width + x) as usize) * 4;
        self.pixels[idx] = color.0;
        self.pixels[idx + 1] = color.1;
        self.pixels[idx + 2] = color.2;
        self.pixels[idx + 3] = color.3;
    }

    /// Draw a filled rectangle with top-left (x, y), width w, height h.
    pub fn fill_rect(&mut self, x: i32, y: i32, w: i32, h: i32, color: Color) {
        if w <= 0 || h <= 0 { return; }
        let x0 = x.max(0) as u32;
        let y0 = y.max(0) as u32;
        let x1 = (x + w).min(self.width as i32) as u32;
        let y1 = (y + h).min(self.height as i32) as u32;
        for yy in y0..y1 {
            let base = (yy * self.width) as usize * 4;
            for xx in x0..x1 {
                let idx = base + (xx as usize) * 4;
                self.pixels[idx] = color.0;
                self.pixels[idx + 1] = color.1;
                self.pixels[idx + 2] = color.2;
                self.pixels[idx + 3] = color.3;
            }
        }
    }

    /// Draw a simple anti-aliased-looking line using integer Bresenham (no AA really).
    pub fn draw_line(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, color: Color) {
        let mut x0 = x0;
        let mut y0 = y0;
        let dx = (x1 - x0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let dy = -(y1 - y0).abs();
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        loop {
            self.put_pixel(x0, y0, color);
            if x0 == x1 && y0 == y1 { break; }
            let e2 = 2 * err;
            if e2 >= dy { err += dy; x0 += sx; }
            if e2 <= dx { err += dx; y0 += sy; }
        }
    }

    pub fn into_render_frame(self) -> RenderFrame {
        RenderFrame::Pixels { width: self.width, height: self.height, data: self.pixels }
    }
}
