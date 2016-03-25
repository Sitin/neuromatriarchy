import java.io.File;
import java.util.List;


PImage img;
PImage background;
PImage footer;
File framesDir;
File [] frames;
File currentFrame;
File lastFrame;

int targetWidth = 1024;
int targetHeight = 768;
int rightShift = 0;

void setup() {
  //fullScreen(P2D, 3);
  frameRate(12);
  size(1024, 768);  // size always goes first!
  surface.setResizable(true);
  background(0);

  rightShift = width - targetWidth;

  framesDir = new File(dataPath("frames"));
  background = loadImage("background.jpg");
  image(background, rightShift, 0);
  footer = background.get(0, targetHeight - 30, targetWidth, targetHeight);
}

void draw() {
  if (lastFrame != null && lastFrame.exists()) {
    lastFrame.delete();
  }

  frames = framesDir.listFiles();

  if (frames.length > 1) {
    lastFrame = frames[0];
    currentFrame = frames[1];

    String framePath = currentFrame.getPath();
    String []nameParts = framePath.split("/");
    String frameName = nameParts[nameParts.length - 1];

    img = loadImage(framePath);
    image(img, rightShift + targetWidth / 2 - 350, 0);

    fill(0, 0, 0);
    image(footer, rightShift, targetHeight - 30);
    fill(128, 128, 128);
    noStroke();
    text(frameName, rightShift + 10, targetHeight - 10);

    println(frameName);
  }
}