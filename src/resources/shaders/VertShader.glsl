#version 400 

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 color;
layout(location = 2) in vec3 inTextCoord;


uniform mat4 perspectiveMatrix;
uniform mat4 modelMatrix;
uniform vec3 vertexNormal;

smooth out vec4 theColor;
out vec3 textCoord;//Coordinate of the texture

vec3 computeLightForVertex(vec3 vertexNormal, vec3 color){
	vec3 ligthVector = vec3(0.0, 100.0, 0.0);//Always same ligth position
	vec3 ligthColor = vec3(1.0,1.0,1.0);//White

	float difuse = .5; //How much of the ligth will be diffused 
	// Formula  color = Light*normal*ligth_color*difuse;
	vec3 newColor = difuse*ligthColor*(ligthVector*vertexNormal);
	
	return newColor;
}

void main(){

    vec4 cameraPos = modelMatrix *  position;
    gl_Position = perspectiveMatrix * cameraPos;

    //gl_Position = position;
    //gl_Position = perspectiveMatrix * modelMatrix * position;//This is what makes the ROI not show
    // We need to understand better the whole matrix and world stuff
    theColor = color;
    textCoord= inTextCoord;
}
