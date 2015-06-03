#version 400

//This one makes the interpolation 'smooth' it can be 'flat' or 'noperspective'
// but it needs to be the same as in the vertex shader
smooth in vec4 theColor;
smooth in vec3 textCoord; //This is the texture coordinate

uniform mat4 perspectiveMatrix;
uniform mat4 modelMatrix;

//This part is for the textures
uniform sampler3D imgSampler;// Used to specify how to apply texture

layout (location = 0 ) out vec4 outputColor;

void main()
{
    vec3 currTextCoord = textCoord;
    vec4 textColor = vec4(0);
    textColor = texture(imgSampler, currTextCoord);

    outputColor = vec4(textColor.r,textColor.r, textColor.r, .1);
}
