#version 400

//This one makes the interpolation 'smooth' it can be 'flat' or 'noperspective'
// but it needs to be the same as in the vertex shader
smooth in vec4 theColor;
smooth in vec3 textCoord; //This is the texture coordinate

uniform mat4 perspectiveMatrix;
uniform mat4 modelMatrix;
uniform int dispSegmentation;
uniform int drawPlanes;

//This part is for the textures
uniform sampler3D imgSampler;// Used to specify how to apply texture
uniform sampler3D segSampler;// Used to specify how to apply texture


layout (location = 0 ) out vec4 outputColor;

void main()
{
    vec3 currTextCoord = textCoord;
    vec4 textColor = vec4(0);
    textColor = texture(imgSampler, currTextCoord);

    float count = 100;
    float gamma = 4;
    float decay = .15;

    textColor.r = textColor.r*decay;
    //textColor.r = textColor.r*(gamma/count);
    outputColor = vec4(textColor.r,textColor.r, textColor.r, .4);

    float th= .001;
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
    float bthreshold = .1;// This is the threshold of the SDF to be displayed in red

    if(drawPlanes == 1){
        vec4 mixVal = vec4(.5,.5, .5, .5); 
        textColor = texture(imgSampler, textCoord);
        outputColor = outputColor + vec4(textColor.r, textColor.r, textColor.r, .5);
        // Move if outside the loop
        if(dispSegmentation == 1){
            textColor = texture(segSampler, currTextCoord);
            if( textColor.r <= bthreshold ){
                outputColor = mix(outputColor,vec4(1, 0, 0, 1),mixVal);
            }else{
                outputColor = mix(outputColor,vec4(0, 0, textColor.r/20, 1),mixVal);
            }
        }
    }else{

        vec4 mixVal = vec4(.05,0.05, 0.05, 0); 
        int firstNonZero= 0;
        for(int i = 1; i < count; i++){
            currTextCoord = currTextCoord + dir;
            if( any(greaterThan(currTextCoord, maxCoords)) || 
                    any(lessThan(currTextCoord, minCoords))  ){
                //break;
            }else{
                //Verify we are not out of bounds
                textColor = texture(imgSampler, currTextCoord);
                if(textColor.r > 0){
                    firstNonZero = i;
                }
                if(firstNonZero > 0){
                    textColor.r = textColor.r*(decay/(i+2-firstNonZero));
                    //textColor.r = textColor.r*(gamma/count);
                    outputColor = outputColor + vec4(textColor.r, textColor.r, textColor.r, 0);
                }
                // Move if outside the loop
                if(dispSegmentation == 1){
                    textColor = texture(segSampler, currTextCoord);
                    if( (textColor.r <= bthreshold) && (textColor.r <= bthreshold)){
                        outputColor = mix(outputColor,vec4(1, 0, 0, 0),mixVal);
                    }
                }
            }
        }
    }
}

