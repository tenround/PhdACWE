#version 400

//This one makes the interpolation 'smooth' it can be 'flat' or 'noperspective'
// but it needs to be the same as in the vertex shader
smooth in vec4 theColor;
layout (location = 0 ) out vec4 outputColor;

//This part is for the textures
in vec3 textCoord; //This is the texture coordinate

uniform sampler3D textSampler;// Used to specify how to apply texture
uniform int defaultColor;

vec4 mergeCircleColorAndTexture(vec4 firstColor, vec3 textCoord, sampler3D textSampler){

	float dx = textCoord.x - 0.5;
	float dy = textCoord.y - 0.5;
	float dist = sqrt(dx*dx + dy*dy);
	//vec4 firstColor = vec4(1.0, 0.0, 0.0, 1.0);
	vec4 secondColor = vec4(0.0, 0.0, 1.0, 1.0);
	float start = 0;
	float end = 1;

	vec4 computedColor = mix(firstColor, secondColor, 
							smoothstep( start, end, dist));

    vec4 halfText= texture(textSampler, textCoord);
	//computedColor = computedColor/2;
    outputColor = vec4(halfText.r,halfText.r,halfText.r+.05,.5);
	return outputColor;
}

void main()
{
	switch(defaultColor){
		case 1:
			outputColor = mergeCircleColorAndTexture(vec4(1.0,0.0,0.0,1.0), textCoord, textSampler);
			break;
		case 2:
			outputColor = mergeCircleColorAndTexture(vec4(0.0,1.0,0.0,1.0), textCoord, textSampler);
			break;
		case 3:
			outputColor = mergeCircleColorAndTexture(vec4(0.0,0.0,1.0,1.0), textCoord, textSampler);
			break;
		default:
			outputColor = mergeCircleColorAndTexture(vec4(1.0,0.0,1.0,1.0), textCoord, textSampler);
			break;
	}
	// Default behaviour, merges color and texture
	//outputColor = defaultMergeColorAndTexture(theColor, textCoord, textSampler);
	//------ Extra example (drawing a circle)--------
}


