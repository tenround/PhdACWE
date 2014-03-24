/**
 * For vectors X and Y it computes SUM of (x.i - y.i)^2     for i != dimension 
 * for the moment only works for dimension 1 and 2
 */
float getVectDistRestricted(int2 a, int2 b, int d){

    float dist;
	if(d == 1){
	   dist = (float)pow((float)a.y-b.y,2);
	}
	else {
	   dist = (float)pow((float)a.x-b.x,2);
	}
    return dist;
}

/**
 * It computes the L2 norm if 'squared' is false and
 * the L2^2 if 'squared' is true
 */
float getVectDist(int2 a, int2 b, bool squared ){
    float2 c = (float2)(a.x-b.x, a.y-b.y);
    float dist;

    if( squared)
        dist =  pow(c.x,2) + pow(c.y,2);
    else
        dist = sqrt( pow(c.x,2) + pow(c.y,2));

    return dist;
}

/**
 * Verifies that the content of two 2D vectors is the same ( a = b )
 */
bool equalv(int2 a, int2 b){
    if( (a.x == b.x) && (a.y == b.y) ){
        return true;
    }else{
        return false;
    }
}

/**
 * Gets the 1D index from a 2D coordinates. (from (row,col) it gives a unique index
 */
int indxFromCoord(int width, int height, int row, int col, int dim){
    return width*row*dim + dim*col + 1;
}

/**
 * It gets the two dimension coordinates from 1D index. From indx it gives the
 * corresponding row and column
 */
int2 getCoords(int indx, int width){

	indx = indx - 1; //To avoid the index 0
    int2 coords = (int2)(0,0);
    coords.x = (int)floor((float) indx/(width) );
    coords.y = indx - coords.x*width;

    return coords;
}

/**
 * It receives  g[l-2] g[l-1] currIndx currcords
 */
bool removeFV(int ui, int vi, int wi, int2 curr_cell, int width, int d){
    int2 u = getCoords(ui,width);
    int2 v = getCoords(vi,width);
    int2 w = getCoords(wi,width);

    float a;
    float b;

    if(d==1){
        a = v.x - u.x;
        b = w.x - v.x;
    }else{
        a = v.y - u.y;
        b = w.y - v.y;
    }

    float c = a + b;

    float test_eqn =	  c*getVectDistRestricted(curr_cell, v, d)
                        - b*getVectDistRestricted(curr_cell, u, d)
                        - a*getVectDistRestricted(curr_cell, w, d)
                        - a*b*c;

    if( test_eqn > 0)
        return true;
    else
        return false;
}

/**
 * This function returns the current 'pseudo row' that we are analyzing. 
 * if we are on dimension 1 then it returns a real row of the image
 * if we are on dimension 2 then it returns a column
 */
int2 getCurrCell(int curr_row, int curr_col, int cell, int d){
    if(d==1){
        return (int2)(cell,curr_col);
    }else{
        return (int2)(curr_row,cell);
    }
}

__kernel void
SDF_voroStep2(read_only image2d_t src, write_only image2d_t dst,
        int width, int height,  sampler_t sampler, int d)
{
	// Maximum possible distance (from one corner to the other, used to normalize
	// the distances from 0 to 255)
	float maxDist = getVectDist( (int2)(0,0), (int2)(width,height), false);

    int curr_col = (int)get_global_id(0);
    int curr_row = (int)get_global_id(1);

    int rwidth = 0;//This will be the line width (dependent of current d)
	int g[2048];// TODO THIS ARRAY SHOULD BE THE SIZE OF MAX(width, height)

    if(d==1){// d=1 It should iterate in the same way as matlab
	//For d==1 internal loop is for rows and for d=2 internal is for columns
        rwidth = height;
    }
    else{
        rwidth = width;
    }

    int l=0;
    float4 x = (float4)(0 , 0 , 0 , 0);
	float4 fij = (float4)(0,0,0,0);
    int2 curr_cell = (int2)(0 , 0);

	//This is the internal loop on the matlab code
    for(int cell=0; cell < rwidth; cell++){

        //This is the current cell we are analyzing for d=1 it iterates over a column
        curr_cell = getCurrCell(curr_row, curr_col, cell, d);

		// Current cell has first row and then col we need to swap it
		x = read_imagef(src, sampler, (int2)(curr_cell.y, curr_cell.x) );

        if( x.x > 0){
            if( l>1 ){
               while( (l>1) && (removeFV( g[l-2] , g[l-1] , x.x , curr_cell ,
                                     width , d) )) {
                    l--;
                }
			}
			g[l] = x.x;
			l++;
        }
    }

	int totg = l-1;// Save total amount
    l = 0;

    int2 g1 = (int2)(0,0);
    int2 g2 = (int2)(0,0);
    bool fastDist = false;// Avoids the squared root

    if( totg >= 0){
        for(int cell=0; cell < rwidth; cell++){
            curr_cell = getCurrCell(curr_row, curr_col, cell, d);

            if(totg > l){

                //Sets the current CFV and the next FV to see which one is closest
                g1 = getCoords(g[l], width);
                g2 = getCoords(g[l+1], width);

                while( (l < totg) &&
                            ( getVectDist( curr_cell , g1, fastDist ) >
                                        getVectDist( curr_cell , g2, fastDist ))){
                    l = l+1;
                    if( totg > l){
                        g1 = getCoords(g[l] , width);
                        g2 = getCoords(g[l+1] , width);
                    }
                }
            }

			fij = (float4)(0,0,0,0);
			if( d == 1){
				fij = (float4)(g[l],0,0,255);
			}
			else{ 
		// Write directly the distance in this case from 0 to max(width,height)
//				fij = max(width,height)*((float4)(getVectDist(curr_cell,
//							getCoords(g[l], width), false) ,0,0,255))/maxDist;

			fij = (float4)(getVectDist(curr_cell,getCoords(g[l], width), false),0,0,255);
//                    fij = (float4)(g[l],0,0,255); //Used to show the closest index
//                    fij = (float4)(totg,0,0,255); //Used to show the closest index
			}

			write_imagef(dst, (int2)(curr_cell.y,curr_cell.x), fij);
		}
    }
    else{
		// If there is no FV in this row leave all the 'row' with 0's
		for(int cell=0; cell < rwidth; cell++){
            curr_cell = getCurrCell(curr_row, curr_col, cell, d);
			write_imagef(dst, (int2)(curr_cell.y,curr_cell.x), (float4)(0,0,0,255));
		}
    }
}


__kernel void
mergePhis( read_only image2d_t img_phi_half,read_only image2d_t img_phi_sec_half,
			write_only image2d_t img_phi,
        sampler_t sampler){
				
	int col = get_global_id(0);
	int row = get_global_id(1);

	float4 fhalf = read_imagef(img_phi_half, sampler, (int2)(col,row));
	float4 sechalf = read_imagef(img_phi_sec_half, sampler, (int2)(col,row));

	if( sechalf.x > 0){
		//The plus 1 is in order to have a band of 0
		write_imagef( img_phi, (int2)(col,row),
					(float4)(fhalf.x-sechalf.x+1, 0,sechalf.x,255));
//					(float4)(0, 0,sechalf.x,255));
	}else{
		write_imagef( img_phi, (int2)(col,row), (float4)(fhalf.x, fhalf.x,0,255));
	}
}


/**
 * Fills a new image with the values of a '1D index' inside all the
 * pixels that has value > 0 if mode == 1 or for all the pixels
 * that has value < 0 if mode == 0
 */
__kernel void
SDF_voroStep1(read_only image2d_t src, write_only image2d_t dst,
        int width, int height, sampler_t sampler, int mode)
{
    int col = (int)get_global_id(0);
    int row = (int)get_global_id(1);

    uint4 x = read_imageui(src, sampler, (int2)(col,row));

	if( mode == 1){ // Distance to values >  0
		if(x.x > 0){
			int currIndx = indxFromCoord(width, height, row,col,1);
			write_imagef(dst, (int2)(col,row), (float4)(currIndx,0,0,255));
		}
		else{
			write_imagef(dst, (int2)(col,row), (float4)(0,0,0,255));
		}
	}

	if(mode == 0){ // Distance to values == 0
		if(x.x == 0){
			int currIndx = indxFromCoord(width, height, row,col,1);
			write_imagef(dst, (int2)(col,row), (float4)(currIndx,0,0,255));
		}
		else{
			write_imagef(dst, (int2)(col,row), (float4)(0,0,0,255));
		}
	}
}

