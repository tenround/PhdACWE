#version 330

//This one makes the interpolation 'smooth' it can be 'flat' or 'noperspective'
// but it needs to be the same as in the vertex shader
smooth in vec4 theColor;
out vec4 outputColor;

//This part is for the textures
in vec2 colorCoord; //This is the texture coordinate
uniform sampler2D textSampler;// Used to specify how to apply texture

void main()
{

    vec4 halfColor = 0.5*theColor;
    vec4 halfText= 0.5*texture(textSampler, colorCoord);
    //outputColor = halfText+halfColor; 
    outputColor = 2*halfText; 

}
