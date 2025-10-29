use crate::core::{GymError, RenderFrame, Result};

/// Encode a RenderFrame::Pixels to a PNG byte vector.
/// - When the `image` feature is enabled, this will encode using the `image` crate.
/// - Without the feature, returns GymError::NotSupported.
pub fn encode_png(frame: &RenderFrame) -> Result<Vec<u8>> {
    match frame {
        RenderFrame::Pixels { width, height, data } => encode_pixels_png(*width, *height, data),
        RenderFrame::Text(_) => Err(GymError::NotSupported("Text frames cannot be encoded to PNG".into())),
    }
}

#[cfg(feature = "image")]
fn encode_pixels_png(width: u32, height: u32, data: &Vec<u8>) -> Result<Vec<u8>> {
    use image::codecs::png::PngEncoder;
    use image::{ColorType, ImageEncoder};
    use std::io::Cursor;

    let pixels = data.as_slice();
    let count = (width as usize) * (height as usize);
    let channels = if pixels.len() == count * 3 {
        3
    } else if pixels.len() == count * 4 {
        4
    } else {
        return Err(GymError::InvalidObservation(format!(
            "Pixel data length {} does not match width*height*3 or *4 ({}x{})",
            pixels.len(), width, height
        )));
    };

    let color = if channels == 3 { ColorType::Rgb8 } else { ColorType::Rgba8 };

    let mut buf = Vec::new();
    {
        let mut cursor = Cursor::new(&mut buf);
        let encoder = PngEncoder::new(&mut cursor);
        encoder
            .write_image(pixels, width, height, color.into())
            .map_err(|e| GymError::Other(format!("PNG encode error: {}", e)))?;
    }
    Ok(buf)
}

#[cfg(not(feature = "image"))]
fn encode_pixels_png(_width: u32, _height: u32, _data: &Vec<u8>) -> Result<Vec<u8>> {
    Err(GymError::NotSupported(
        "PNG encoding requires the `image` feature".into(),
    ))
}

/// Save a RenderFrame::Pixels as a PNG file at the given path.
/// Requires the `image` feature; otherwise returns NotSupported.
pub fn save_png<P: AsRef<std::path::Path>>(path: P, frame: &RenderFrame) -> Result<()> {
    let bytes = encode_png(frame)?;
    std::fs::write(path, bytes).map_err(|e| GymError::Other(format!("Failed to write PNG: {}", e)))
}
