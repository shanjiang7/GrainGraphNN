import os
from moviepy.editor import ImageSequenceClip

# Folder where your images are stored.
image_folder = 'overlap_plots'
# List all PNG files in the folder, sorted by filename.
image_files = sorted([os.path.join(image_folder, img)
                      for img in os.listdir(image_folder)
                      if img.endswith(".png")])

# Create a video clip from the image sequence.
clip = ImageSequenceClip(image_files, fps=2)

# Write the video file.
clip.write_videofile("evolution_1.mp4", codec="libx264")

image_folder_2 = 'class_before'
# List all PNG files in the folder, sorted by filename.
image_files = sorted([os.path.join(image_folder_2, img)
                      for img in os.listdir(image_folder_2)
                      if img.endswith(".png")])

# Create a video clip from the image sequence.
clip = ImageSequenceClip(image_files, fps=2)

# Write the video file.
clip.write_videofile("evolution_2.mp4", codec="libx264")