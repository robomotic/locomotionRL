#!/usr/bin/env python3
"""
Convert MP4 videos to animated GIFs.
Supports batch conversion and quality/size optimization.
"""
import os
import argparse
from pathlib import Path
from moviepy.editor import VideoFileClip

def convert_mp4_to_gif(input_path, output_path=None, fps=10, scale=1.0, optimize=True):
    """
    Convert an MP4 video to an animated GIF.
    
    Args:
        input_path: Path to the input MP4 file
        output_path: Path for the output GIF (if None, uses same name as input)
        fps: Frames per second for the GIF (lower = smaller file)
        scale: Scale factor for resizing (0.5 = half size, 1.0 = original)
        optimize: Whether to optimize the GIF (reduces file size)
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return False
    
    # Generate output path if not provided
    if output_path is None:
        output_path = input_path.with_suffix('.gif')
    else:
        output_path = Path(output_path)
    
    print(f"Converting: {input_path.name}")
    print(f"  → {output_path.name}")
    
    try:
        # Load the video
        clip = VideoFileClip(str(input_path))
        
        # Resize if needed
        if scale != 1.0:
            clip = clip.resize(scale)
            print(f"  Resizing to {int(clip.w)}x{int(clip.h)} ({scale*100:.0f}%)")
        
        # Write GIF
        print(f"  Writing GIF at {fps} fps...")
        clip.write_gif(
            str(output_path),
            fps=fps,
            program='ffmpeg',  # Use ffmpeg for better quality
            opt='nq' if optimize else None  # 'nq' = no quantization optimization
        )
        
        clip.close()
        
        # Show file sizes
        input_size = input_path.stat().st_size / (1024 * 1024)  # MB
        output_size = output_path.stat().st_size / (1024 * 1024)  # MB
        
        print(f"  ✓ Done! {input_size:.2f} MB → {output_size:.2f} MB")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def convert_directory(directory, fps=10, scale=1.0, optimize=True, pattern="*.mp4"):
    """
    Convert all MP4 files in a directory to GIFs.
    
    Args:
        directory: Directory containing MP4 files
        fps: Frames per second for GIFs
        scale: Scale factor for resizing
        optimize: Whether to optimize GIFs
        pattern: File pattern to match (default: *.mp4)
    """
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        return
    
    mp4_files = list(directory.glob(pattern))
    
    if not mp4_files:
        print(f"No MP4 files found in {directory}")
        return
    
    print(f"Found {len(mp4_files)} MP4 file(s) to convert\n")
    
    success_count = 0
    for mp4_file in mp4_files:
        if convert_mp4_to_gif(mp4_file, fps=fps, scale=scale, optimize=optimize):
            success_count += 1
        print()  # Blank line between conversions
    
    print(f"Conversion complete: {success_count}/{len(mp4_files)} successful")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert MP4 videos to animated GIFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single video
  python convert_to_gif.py video.mp4
  
  # Convert with custom settings
  python convert_to_gif.py video.mp4 --fps 15 --scale 0.75
  
  # Convert all videos in a directory
  python convert_to_gif.py --dir ./videos/
  
  # Convert with specific output name
  python convert_to_gif.py video.mp4 --output my_animation.gif
        """
    )
    
    parser.add_argument("input", nargs="?", help="Input MP4 file")
    parser.add_argument("--output", "-o", help="Output GIF file (default: same name as input)")
    parser.add_argument("--dir", "-d", help="Convert all MP4 files in this directory")
    parser.add_argument("--fps", type=int, default=10, 
                        help="Frames per second (default: 10, lower = smaller file)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale factor for resizing (default: 1.0, e.g., 0.5 = half size)")
    parser.add_argument("--no-optimize", action="store_true",
                        help="Disable GIF optimization (faster but larger files)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.dir:
        # Directory mode
        convert_directory(
            args.dir,
            fps=args.fps,
            scale=args.scale,
            optimize=not args.no_optimize
        )
    elif args.input:
        # Single file mode
        convert_mp4_to_gif(
            args.input,
            output_path=args.output,
            fps=args.fps,
            scale=args.scale,
            optimize=not args.no_optimize
        )
    else:
        parser.print_help()
        print("\nError: Please provide either an input file or --dir argument")
