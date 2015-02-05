#version 400

//This one makes the interpolation 'smooth' it can be 'flat' or 'noperspective'
// but it needs to be the same as in the vertex shader
smooth in vec4 theColor;
layout (location = 0 ) out vec4 outputColor;

//This part is for the textures
in vec3 textCoord; //This is the texture coordinate

uniform sampler3D textSampler;// Used to specify how to apply texture

void main()
{
    vec3 currTextCoord = textCoord;
    vec4 textColor = texture(textSampler, currTextCoord);
    vec4 tempcolor = vec4(0);
    outputColor = vec4(textColor.r,textColor.r,textColor.r,1);

    for(int i = 1; i < 100; i++){
        currTextCoord.z = currTextCoord.z + i/200;
        tempcolor = texture(textSampler, currTextCoord);
        outputColor.x = outputColor.x + tempcolor;
    }
}


