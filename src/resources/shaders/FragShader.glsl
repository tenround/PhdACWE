#version 400

//This one makes the interpolation 'smooth' it can be 'flat' or 'noperspective'
// but it needs to be the same as in the vertex shader
smooth in vec4 theColor;
layout (location = 0 ) out vec4 outputColor;

//This part is for the textures
in vec2 textCoord; //This is the texture coordinate
uniform sampler2D textSampler;// Used to specify how to apply texture

vec4 defaultMergeColorAndTexture(vec4 theColor, vec2 textCoord, sampler2D textSampler){
    vec4 halfColor = 0.5*theColor;
    vec4 halfText= 0.5*texture(textSampler, textCoord);
    outputColor = halfText+halfColor; 
    //outputColor = 2*halfText; 
	return outputColor;
}


vec4 mergeCircleColorAndTexture(vec4 theColor, vec2 textCoord, sampler2D textSampler){

	float dx = textCoord.x - 0.5;
	float dy = textCoord.y - 0.5;
	float dist = sqrt(dx*dx + dy*dy);
	vec4 firstColor = vec4(1.0, 0.0, 0.0, 1.0);
	vec4 secondColor = vec4(0.0, 0.0, 1.0, 1.0);
	float start = 0;
	float end = 1;

	vec4 computedColor = mix(firstColor, secondColor, 
							smoothstep( start, end, dist));

    vec4 halfText= 0.5*texture(textSampler, textCoord);
	computedColor = computedColor/2;
    outputColor = halfText+computedColor; 
	return outputColor;
}

void main()
{
    vec4 halfColor = 0.5*theColor;
    vec4 halfText= 0.5*texture(textSampler, textCoord);
    outputColor = halfText+halfColor; 
    //outputColor = 2*halfText; 

	// Default behaviour, merges color and texture
	//outputColor = defaultMergeColorAndTexture(theColor, textCoord, textSampler);

	//------ Extra example (drawing a circle)--------
	outputColor = mergeCircleColorAndTexture(theColor, textCoord, textSampler);
}


