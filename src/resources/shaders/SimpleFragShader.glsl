#version 400

//This one makes the interpolation 'smooth' it can be 'flat' or 'noperspective'
// but it needs to be the same as in the vertex shader
smooth in vec4 theColor;
smooth in vec3 textCoord; //This is the texture coordinate

uniform mat4 perspectiveMatrix;
uniform mat4 modelMatrix;

//This part is for the textures
uniform sampler3D textSampler;// Used to specify how to apply texture

layout (location = 0 ) out vec4 outputColor;

void main()
{
    vec3 currTextCoord = textCoord;
    vec4 textColor = vec4(0);
    textColor = texture(textSampler, currTextCoord);
    float count = 100;
    float gamma = 3;
    float decay = .5;
    textColor.r = textColor.r*decay;
    //textColor.r = textColor.r*(gamma/count);
    outputColor = vec4(0,0,0,.1);

    float th= .1;
    //Directions are: (+R-L,+Near-Far,+Down-Up) but are been swapped below
    vec3 dir = vec3(0,0,-1/(count));
    mat4 swapMat= mat4(
            1.0, 0.0, 0.0, 0.0, // first column 
            0.0, 0.0, -1.0, 0.0, // second column
            0.0, 1.0, 0.0, 0.0, // third column
            0.0, 0.0, 0.0, 0.0);

    mat4 tempMat = swapMat * modelMatrix;

    vec4 finDir = tempMat * vec4(dir,1);

    vec3 maxCoords = vec3(1+th,1+th,1+th);
    vec3 minCoords=  vec3(-th,-th,-th);

    dir= finDir.xyz;
    for(int i = 1; i < count; i++){
        currTextCoord = currTextCoord + dir;
        if( any(greaterThan(currTextCoord, maxCoords)) || 
                any(lessThan(currTextCoord, minCoords))  ){
            //break;
        }else{
            //Verify we are not out of bounds
            textColor = texture(textSampler, currTextCoord);
            textColor.r = textColor.r*(decay/i);
            //textColor.r = textColor.r*(gamma/count);
            if(   textColor.r <= th && textColor.r >= -th ){
                outputColor = outputColor + vec4(.1, 0, 0, 0);
            }
        }
    }
}

