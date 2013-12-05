#version 330

//This one makes the interpolation 'smooth' it can be 'flat' or 'noperspective'
// but it needs to be the same as in the vertex shader
//smooth in vec4 theColor;
out vec4 outputColor;

void main()
{
    //outputColor = theColor; 
    outputColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
}
