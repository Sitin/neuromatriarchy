import java.io.File;
import java.util.List;


PImage img;
File framesDir;
File [] frames;
File currentFrame;
File lastFrame;

void setup() {
  //fullScreen();
  frameRate(12);
  size(800, 800);  // size always goes first!
  surface.setResizable(true);
  background(0);
  
  framesDir = new File(dataPath("frames"));
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
    image(img, 0, 0);
    
    fill(128, 128, 128);
    noStroke();
    text(frameName, 10, height - 50);
    
    println(frameName);
  }
}