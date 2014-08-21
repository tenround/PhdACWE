#version 400 

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 color;
layout(location = 2) in vec2 inTextCoord;


uniform mat4 perspectiveMatrix;
uniform mat4 modelMatrix;

smooth out vec4 theColor;

out vec2 textCoord;//Coordinate of the texture

void main(){

    vec4 cameraPos = modelMatrix *  position;
    gl_Position = perspectiveMatrix * cameraPos;

    //gl_Position = position;
    //gl_Position = perspectiveMatrix * modelMatrix * position;//This is what makes the ROI not show
    // We need to understand better the whole matrix and world stuff
    theColor = color;
    textCoord= inTextCoord;
}
