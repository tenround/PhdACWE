#define MAXF 1
#define MAXDPHIDT 2


__constant sampler_t def_sampler = CLK_NORMALIZED_COORDS_FALSE |
								CLK_ADDRESS_CLAMP_TO_EDGE |
								CLK_FILTER_NEAREST;
__constant float thres = 2;
__constant float EPS = .00001;

// It computes the curvature of the curve phi close to 0 and
// also the value of the Force F
__kernel void
smoothPhi(__global float* img_phi, __global float* img_smooth_phi,
					float alpha){

	sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
								CLK_ADDRESS_REPEAT|
								CLK_FILTER_NEAREST;

	int width = get_image_width(img_phi);
	int height = get_image_height(img_phi);

    int col = (int)get_global_id(0);
    int row = (int)get_global_id(1);

	float4 phi= read_imagef(img_phi, sampler, (int2)(col,row));

	// Compute Curvature
	float4 shiftR = read_imagef(img_phi, sampler, (int2)(col-1,row)); 
	float4 shiftL = read_imagef(img_phi, sampler, (int2)(col+1,row)); 
	float4 shiftD = read_imagef(img_phi, sampler, (int2)(col,row+1)); 
	float4 shiftU = read_imagef(img_phi, sampler, (int2)(col,row-1)); 

	float4 a = phi - shiftR;
	float4 b = shiftL - phi;
	float4 c = phi - shiftD;
	float4 d = shiftU - phi;

	float4 an = a;
	float4 bn = b;
	float4 cn = c;
	float4 dn = d;

	float4 ap = a;
	float4 bp = b;
	float4 cp = c;
	float4 dp = d;

	ap.x = a.x < 0? 0: a.x;
	bp.x = b.x < 0? 0: b.x;
	cp.x = c.x < 0? 0: c.x;
	dp.x = d.x < 0? 0: d.x;

	an.x = a.x > 0? 0: a.x;
	bn.x = b.x > 0? 0: b.x;
	cn.x = c.x > 0? 0: c.x;
	dn.x = d.x > 0? 0: d.x;


	float dD = 0;
	if( phi.x > 0){
		dD = sqrt( max( pow(ap.x,2), pow(bn.x,2)) + max( pow(cp.x,2), pow( dn.x,2)) ) -1;
	}
	if( phi.x < 0){
		dD = sqrt( max( pow(an.x,2), pow(bp.x,2)) + max( pow(cn.x,2), pow( dp.x,2)) ) -1;
	}

	float newVal = phi.x - alpha* (phi.x/sqrt( pow(phi.x,2) + 1))*dD;


	float contour = 0;
	if( (newVal <= thres) && (newVal >= -thres) ){
		contour = 255;
	}

	write_imagef(img_smooth_phi, (int2)(col,row), (float4)(newVal,contour,0,255));
}

__kernel
void newphi( read_only image2d_t img_phi, read_only image2d_t img_dphidt, 
			  write_only image2d_t img_newphi,  sampler_t sampler,
			global float* max_dphidt){
				
	float dt = .45/(max_dphidt[0] + EPS);
	int col = get_global_id(0);
	int row = get_global_id(1);

	float4 phi = read_imagef(img_phi, sampler, (int2)(col,row));
	float4 dphidt= read_imagef(img_dphidt, sampler, (int2)(col,row));

	float4 newval = phi + dt*dphidt;
	
	write_imagef(img_newphi, (int2)(col,row), newval);
}

/**
	* Gets the maximum value for each warp Integers
	* The indexes should always be in one dimension
*/
__kernel
void ArrMaxFloat( global float* arr_in, int block, int length, int use_abs) {

    int glob_indx = (int)get_global_id(0)*block;
	int upper_bound = min(glob_indx + block, length);

	int grp_indx = get_global_id(0);

	// Shared variables among the local threads
	float max = -1000;
	float value = 0; 

	// Reads the current value
	while(glob_indx < upper_bound){
		value = arr_in[glob_indx];

		if(use_abs){ value = fabs( (float) value); }

		if(value > max){
			max = value;
		}
		glob_indx ++;
	}

	arr_in[grp_indx] = max;
}

/**
	* Gets the maximum value for each warp Integers
	* The indexes should always be in one dimension
*/
__kernel
void ArrMaxInt( global int* arr_in, int use_abs ) {
    int col = (int)get_global_id(0);

	int grp_indx = (get_group_id(1) * get_num_groups(0) + get_group_id(0));

	// Shared variables among the local threads
	local int max;

	// Only first thread initializes the local variables
	if( get_local_id(0) == 0 && get_local_id(1) == 0){
		max = 0;
	}	

	// Reads the current value
	float value = arr_in[col];
	if(use_abs){
		value = fabs( (float) value);
	}

	barrier(CLK_LOCAL_MEM_FENCE);

		// Gets the maximum 
		atomic_max(&max, value);

	barrier(CLK_LOCAL_MEM_FENCE);

	// Only the first thread of each group computes the final averages
	if( get_local_id(0) == 0 && get_local_id(1) == 0){
		arr_in[grp_indx] = max;
	}
}

/**
 * This kernel obtains the maximum value for each column in an image
 * The size of tempValues should always be the height of the image
*/
__kernel
void ImgColMax( read_only image2d_t img, __local float* tempValues,
			__global float* result, sampler_t sampler, int layer, int use_abs) {

	int col = get_global_id(0);
	int group = get_group_id(0);
	int height = get_image_height(img);
	int width = get_image_width(img);
	int num_of_threads = get_local_size(0);

	float value = 0;
	float colMax = 0;

	if( col < width){
		for(int row=0; row < height; row++){
			float4 curr_pix = read_imagef(img, sampler, (int2)(col,row));
//			uint4 curr_pix = read_imageui(img, sampler, (int2)(col,row));
			value = 0;
			switch(layer){
				case 1:
					value = curr_pix.x;
					break;
				case 2:
					value = curr_pix.y;
					break;
				case 3:
					value = curr_pix.z;
					break;
			}
			if(use_abs){
				value = fabs( (float) value);
			}
			if( value > colMax){
				colMax = value;
			}
		}

		tempValues[ get_local_id(0) ] = colMax;	
		
		//Until here we have the maxium number of each column
		barrier(CLK_LOCAL_MEM_FENCE);
		
		//Now we get the maximum number of each 'Group'
		if( get_local_id(0) == 0){

			float groupMax = 0;
			for(int i=0; i < num_of_threads; i++){
				if( tempValues[ i ] > groupMax){
					groupMax = tempValues[i];
				}
			}
			result[group] = groupMax;
		}
	}
}

__kernel void
dphidt(read_only image2d_t phi, read_only image2d_t curvAndF, 
		write_only image2d_t dphidt, global float* buf_F, sampler_t sampler, float alpha){

	float maxF = buf_F[0];

	int width = get_image_width(curvAndF);
	int height = get_image_height(curvAndF);

	int col = (int)get_global_id(0);
	int row = (int)get_global_id(1);

	float4 curr_pix = read_imagef(phi, sampler, (int2)(col,row));
	float4 currCurvAndF= read_imagef(curvAndF, sampler, (int2)(col,row));

	//The value of the curvature is on 'x' and the value of F is on 'y'
	float curv = currCurvAndF.x;
	float F = currCurvAndF.y;

	float4 newval = (float4)(0,0,0,255);
	if( (curr_pix.x <= thres) && (curr_pix.x >= -thres) ){
//		newval.x = -255*(F/maxF + alpha*curv); //Usefull to see something in the image
		newval.x = F/(fabs(maxF)) + alpha*curv;
		newval.y = 255;//Usufull to see the border of the curve
	}

	write_imagef(dphidt, (int2)(col,row), newval);
}


// It computes the curvature of the curve phi close to 0 and
// also the value of the Force F
__kernel void
CurvatureAndF(read_only image2d_t phi, read_only image2d_t in,
				write_only image2d_t curvAndF, sampler_t sampler,
						global float* avgDistInOut){

	float avgIn = avgDistInOut[0];
	float avgOut = avgDistInOut[1];

	int width = get_image_width(phi);
	int height = get_image_height(phi);

    int col = (int)get_global_id(0);
    int row = (int)get_global_id(1);

	float4 curr_phi = read_imagef(phi, sampler, (int2)(col,row));
	uint4 curr_img   = read_imageui(in, sampler, (int2)(col,row));

	if( (curr_phi.x <= thres) && (curr_phi.x >= -thres) ){
		// Compute Curvature
		float4 cu	= curr_phi;
		float4 up	= read_imagef(phi, sampler, (int2)(col,row-1));
		float4 ul	= read_imagef(phi, sampler, (int2)(col-1,row-1));
		float4 lf	= read_imagef(phi, sampler, (int2)(col-1,row));
		float4 dl	= read_imagef(phi, sampler, (int2)(col-1,row+1));
		float4 dn	= read_imagef(phi, sampler, (int2)(col,row+1));
		float4 dr	= read_imagef(phi, sampler, (int2)(col,row+1));
		float4 rh	= read_imagef(phi, sampler, (int2)(col+1,row));
		float4 ur	= read_imagef(phi, sampler, (int2)(col+1,row-1));

		float phi_x = -lf.x + rh.x;
		float phi_y = dn.x + up.x;
		float phi_xx = lf.x - 2*cu.x + rh.x;
		float phi_yy = dn.x - 2*cu.x + up.x;
		float phi_xy = -.25*dl.x -.25*ur.x + .25*dr.x + .25*ul.x;
		float phi_x2 = phi_x * phi_x;
		float phi_y2 = phi_y * phi_y;

		float eps = .001;
		float curv = ( phi_x2* phi_yy + phi_y2*phi_xx - 2*phi_x*phi_y*phi_xy)/
					( powr(phi_x2 + phi_y2 + eps , 3/2)*sqrt(phi_x2 + phi_y2) );
		
		curv = 0;

		// Computing F
		float F = pow( (float)curr_img.x - avgIn, 2) -
								pow( (float)curr_img.x - avgOut, 2);
//		F = -F; // Just if you want to search for light places rather than dark places

		float4 curv_and_F_value = (float4)(curv, F, 255, 255);
//		float4 curv_and_F_value = (float4)(0, 255, 0, 255);
		write_imagef(curvAndF, (int2)(col,row), curv_and_F_value);
	}
	else{
		float4 newcurv = (float4)(0, 0, 0, 255);
		write_imagef(curvAndF, (int2)(col,row), newcurv);
	}
}



/**
	* This kernel computes local averages of pixels inside and outside the object
	* for every warp size. It only works for positive values 
*/
__kernel void
Step1AvgInOut(read_only image2d_t phi, read_only image2d_t in, 
				global float* avgDistInOut, global int* avgCount) {

	int width = get_image_width(in);
	int height = get_image_height(in);

    int col = (int)get_global_id(0);
    int row = (int)get_global_id(1);

	int grp_indx = (get_group_id(1) * get_num_groups(0) + get_group_id(0));

	// We will double the group size, setting on the even indexes the inside values
	// and on the odd indexes the out values
	int indxIn = grp_indx*2;
	int indxOut = grp_indx*2 + 1;

	// Shared variables among the local threads
	local int sumIn;
	local int sumOut;
	local int countIn;
	local int countOut;

	// Only first thread initializes the local variables
	if( get_local_id(0) == 0 && get_local_id(1) == 0){
		sumIn = 0;
		sumOut = 0;
		countIn = 0;
		countOut = 0;
	}	

	// Read the image values
	float4 phiPix = read_imagef(phi, def_sampler, (int2)(col,row));
	uint4 inPix = read_imageui(in, def_sampler, (int2)(col,row));

	// TODO ask in what band we want to compute the average (R G or B)
	int value = (int) inPix.x;

	barrier(CLK_LOCAL_MEM_FENCE);

	// Adds the value to the positive or negative index
	if(phiPix.x <= 0){
		// This are pixels inside the mask
		atomic_add(&sumIn, value);
		atomic_inc(&countIn);
	}else{
		// This are pixels ouside the mask
		atomic_add(&sumOut, value);
		atomic_inc(&countOut);
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// Only the first thread of each group computes the final averages
	if( get_local_id(0) == 0 && get_local_id(1) == 0){
		avgDistInOut[indxIn] = (float)sumIn; 
		avgDistInOut[indxOut] = (float)sumOut; 
	}
	// Only thread 2 writes the counts
	if( get_local_id(0) == 1 && get_local_id(1) == 0){
		avgCount[indxIn] = countIn;
		avgCount[indxOut] = countOut;
	}
}
/**
	* Obtains the average value of an array. For this specific case
	* we are computing two different averages, oner for the odd
	* values and one for the even valus of the array.
*/
__kernel void
Step2AvgInOut(global float* avgDistInOut, global int* avgCount) {

    int col = (int)get_global_id(0);

	int grp_indx = (get_group_id(1) * get_num_groups(0) + get_group_id(0));

	// We will double the group size, setting on the even indexes the inside values
	// and on the odd indexes the out values
	int indxIn = grp_indx*2;
	int indxOut = grp_indx*2 + 1;

	// Shared variables among the local threads
	local int sumIn;
	local int sumOut;
	local int countIn;
	local int countOut;

	// Only first thread initializes the local variables
	if( get_local_id(0) == 0 && get_local_id(1) == 0){
		sumIn = 0;
		sumOut = 0;
		countIn = 0;
		countOut = 0;
	}	

	// Reads the average value
	float value = avgDistInOut[col];
	int count = avgCount[col];

	barrier(CLK_LOCAL_MEM_FENCE);

	// Adds the value to the positive or negative index
	if(count > 0){
		if( fmod((float)col,(float)2) == 0){//If it is an even index then it belongs to the sumIn values
			atomic_add(&sumIn, value);
			atomic_add(&countIn, count);
		}else{// If it is an odd index, then it belongs to the avg out values.
			atomic_add(&sumOut, value);
			atomic_add(&countOut, count);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// Only the first thread of each group computes the final averages
	if( get_local_id(0) == 0 && get_local_id(1) == 0){
		if( (get_num_groups(0) == 1) && (get_num_groups(1) == 1) ){ // In this case this is the last iteation

			if(countIn > 0){ avgDistInOut[indxIn] = (float)sumIn/countIn; }
			else{avgDistInOut[indxIn] = 0; }

			if(countOut > 0){ avgDistInOut[indxOut] = (float)sumOut/countOut; }
			else{avgDistInOut[indxOut] = 0; }

		}else{// Normal iteration, not the last one
			if(countIn > 0){ avgDistInOut[indxIn] = (float)sumIn; }
			else{avgDistInOut[indxIn] = 0; }

			if(countOut > 0){ avgDistInOut[indxOut] = (float)sumOut; }
			else{avgDistInOut[indxIn] = 0; }

		}
	}

	// Only the second thread adds to the counts
	if( get_local_id(0) == 1 && get_local_id(1) == 0){

		if( (get_num_groups(0) == 1) && (get_num_groups(1) == 1) ){ // In this case this is the last iteation
			if(countIn > 0){ avgCount[indxIn] = 1; }//We already have the average
			if(countOut > 0){ avgCount[indxOut] = 1; }
		}else{
			avgCount[indxIn] = countIn; 
			avgCount[indxOut] = countOut;
		}
	}

}
