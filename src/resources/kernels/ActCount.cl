//#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

#define MAXF 1
#define MAXDPHIDT 2

__constant sampler_t def_sampler = CLK_NORMALIZED_COORDS_FALSE |
								CLK_ADDRESS_CLAMP_TO_EDGE |
								CLK_FILTER_NEAREST;

__constant float thresporc = .004f;
__constant float EPS = .000000000001f;


float temp_dD(float a, float b, float c, float d, float e, float f, float phi){

    float ap = a < 0? 0: a;
    float bp = b < 0? 0: b;
    float cp = c < 0? 0: c;
    float dp = d < 0? 0: d;
    float ep = e < 0? 0: e;
    float fp = f < 0? 0: f;

    float an = a > 0? 0: a;
    float bn = b > 0? 0: b;
    float cn = c > 0? 0: c;
    float dn = d > 0? 0: d;
    float en = e > 0? 0: e;
    float fn = f > 0? 0: f;

    float dD = 0;
    if( phi > 0){
        dD = sqrt( max( pow(ap,2), pow(bn,2) ) + 
                   max( pow(cp,2), pow(dn,2) ) +
                   max( pow(ep,2), pow(fn,2) ) ) -1;
    }
    if( phi < 0){
        dD = sqrt( max( pow(an,2), pow(bp,2) ) + 
                   max( pow(cn,2), pow(dp,2) ) +
                   max( pow(en,2), pow(fp,2) ) ) -1;
    }
    return dD;
}

float16 temp_dD_vec(float16 a, float16 b, float16 c, float16 d, float16 e, float16 f, float16 phi){

    float16 result;

    float16 ap = a < 0? 0: a;
    float16 bp = b < 0? 0: b;
    float16 cp = c < 0? 0: c;
    float16 dp = d < 0? 0: d;
    float16 ep = e < 0? 0: e;
    float16 fp = f < 0? 0: f;

    float16 an = a > 0? 0: a;
    float16 bn = b > 0? 0: b;
    float16 cn = c > 0? 0: c;
    float16 dn = d > 0? 0: d;
    float16 en = e > 0? 0: e;
    float16 fn = f > 0? 0: f;

    float16 dD = 0;

    float16 opt1 = sqrt( max( pow(ap,2), pow(bn,2) ) + 
                         max( pow(cp,2), pow(dn,2) ) +
                         max( pow(ep,2), pow(fn,2) ) ) -1;

    float16 opt2 = sqrt( max( pow(an,2), pow(bp,2) ) + 
                         max( pow(cn,2), pow(dp,2) ) +
                         max( pow(en,2), pow(fp,2) ) ) -1;
    
    dD = phi > 0 ? opt1 : 0;
    dD = phi < 0 ? opt2 : 0;

    return result;
}

// Forces the Sussman smooth function, smooths the function depending on the value of the norm of gradient
__kernel void
smoothPhiVec(global float* phi, global float* sm_phi, float beta, int width, int height, int depth){

    int globId = (int)get_global_id(0);// From 0 to height*depth
    int localId = get_local_id(0);

    //Obtain current index
    int slice = width*height;//This is the size of one 'slice'
    int row = width;//This is the size of one 'row'

    if(globId < slice){
        int curr = globId*width;//Current value 0, width, 2*width, .. -> init 0 row, init 1 row, init 2 row

        //(test if is last row)
        bool isLastRow = ( (curr+row) % slice == 0) ? true : false;
        //(test if is first row)
        bool isFirstRow =  (curr % slice == 0)? true : false;

        int currVect = globId*width/16;

        int dn = isLastRow? 0: row; //down value 
        int up = isFirstRow? 0: -row; //up value 

        // ----------------- If we are are in the last slice then we can't have a far slice
        if( curr > (slice*(depth-1) - 1) ){ slice = 0;}
        int fcurr = slice;//(far current) 
        int fdn = isLastRow? fcurr : fcurr+row; //far down value 
        int fup = isFirstRow? fcurr : fcurr-row;//far up value

        // ----------------- If we are are in the first slice then we can't have a closer slice
        slice = width*height;
        if( curr < slice ){ slice = 0;}
        // Closer 9 neighbors
        int ccurr = -slice;//(closer current) 
        int cdn = isLastRow? ccurr : ccurr+row; //close down value 
        int cup = isFirstRow? ccurr : ccurr-row; //Close up value

        float16 a,b,c,d,e,f, dD;

        for(int col = 0; col < (width/16); col++){
            // Read all the data
            // Same slice values 
            float16 currvec = vload16(currVect+col,phi );//Current
            float16 lfvec = vload16(currVect+col,phi - 1);//left
            float16 rivec = vload16(currVect+col,phi + 1);//right
            float16 dnvec = vload16(currVect+col,phi + dn); //down value 
            float16 upvec = vload16(currVect+col,phi + up);//up value 
            //Far values 
            float16 fcurrvec = vload16(currVect+col,phi + fcurr);//(far current) 
            // Closer values
            float16 ccurrvec = vload16(currVect+col,phi + ccurr);//(closer current) 

            // Fix the boundaries
            if(col == 0){//Fix all the left values
                lfvec.s0 = lfvec.s1;
            }
            if(col == (width/16) -1){//Fix all the left values
                rivec.sF = rivec.sE;
            }

            a = currvec - lfvec; // Backward in x 
            b = rivec - currvec; // Forward in x 
            c = currvec - upvec; // Backward in y 
            d = dnvec -  currvec; //Forward in y 
            e = currvec - ccurrvec; // Backward in z 
            f = fcurrvec  - currvec; // Forward in z 

            dD = temp_dD_vec(a,b,c,d,e,f, currvec);

            float16 sm_phiVal = currvec - beta * (currvec/sqrt( pow(currvec,2) + 1)) * dD;

            //vstore16(sm_phiVal, currVect+col, sm_phi);
            vstore16(currvec, currVect+col, sm_phi);
        }
    }
}

// It smooths the level set depending on the slope
// Forces the Sussman smooth function, smooths the function depending on the value of the norm of gradient
__kernel void
smoothPhi(global float* phi, global float* sm_phi, float beta, int width, int height, int depth){

    int globId = (int)get_global_id(0);// From 0 to height*depth

    //Obtain current index
    int slice = width*height;//This is the size of one 'slice'
    int row = width;//This is the size of one 'row'

    int curr = globId*width;//Current value 0, width, 2*width, .. -> init 0 row, init 1 row, init 2 row

    //(test if is last row)
    bool isLastRow = ( (curr+row) % slice == 0) ? true : false;
    //(test if is first row)
    bool isFirstRow =  (curr % slice == 0)? true : false;

    // First 8 neighbors same slice
    int lf = curr-1;//The value left  (left)
    int ri = curr+1;//The value right (right) 
    int dn = isLastRow? curr : curr+row; //down value 
    int up = isFirstRow? curr : curr-row; //up value 
    // Farther 9 neighbors

    // ----------------- If we are are in the last slice then we can't have a far slice
    if( curr > (slice*(depth-1) - 1 ) ){ slice = 0;}

    int fcurr = curr + slice;//(far current) 
    int fdn = isLastRow? fcurr : fcurr+row; //far down value 
    int fup = isFirstRow? fcurr : fcurr-row;//far up value

    // ----------------- If we are are in the first slice then we can't have a closer slice
    slice = width*height;
    if( curr < slice ){ slice = 0;}

    // Closer 9 neighbors
    int ccurr = curr - slice;//(closer current) 

    //------------ First column ---------
    float a = 0; // Backward in x Good
    float b = phi[ri] - phi[curr]; // Forward in x Good
    float c = phi[curr] - phi[up]; // Backward in y Good
    float d = phi[dn] - phi[curr]; //Forward in y Good
    float e = phi[curr] - phi[ccurr]; // Backward in z Good
    float f = phi[fcurr] - phi[curr]; // Forward in z Good

    float dD = temp_dD(a,b,c,d,e,f,phi[curr]);
    sm_phi[curr] = phi[curr] - beta * (phi[curr]/sqrt( pow(phi[curr],2) + 1)) * dD;

    //Iterate over the 'middle' columns
    for(int col = 1; col < width-1; col++){
        a = phi[curr+col] - phi[lf+col]; // Backward in x Good
        b = phi[ri+col] - phi[curr+col]; // Forward in x Good
        c = phi[curr+col] - phi[up+col]; // Backward in y Good
        d = phi[dn+col] - phi[curr+col]; //Forward in y Good
        e = phi[curr+col] - phi[ccurr+col]; // Backward in z Good
        f = phi[fcurr+col] - phi[curr+col]; // Forward in z Good

        dD = temp_dD(a,b,c,d,e,f,phi[curr+col]);
        sm_phi[curr+col] = phi[curr+col] - beta * (phi[curr+col]/sqrt( pow(phi[curr+col],2) + 1)) * dD;
    }

    int col = width-1;
    a = phi[curr+col] - phi[lf+col]; // Backward in x Good
    b = 0; // Forward in x Good
    c = phi[curr+col] - phi[up+col]; // Backward in y Good
    d = phi[dn+col] - phi[curr+col]; //Forward in y Good
    e = phi[curr+col] - phi[ccurr+col]; // Backward in z Good
    f = phi[fcurr+col] - phi[curr+col]; // Forward in z Good
    
    dD = temp_dD(a,b,c,d,e,f,phi[curr]);
    sm_phi[curr+col] = phi[curr+col] - beta * (phi[curr+col]/sqrt( pow(phi[curr+col],2) + 1)) * dD;
}


// Forces the Sussman smooth function, smooths the function depending on the value of the norm of gradient
__kernel void
smoothPhiLocal(global float* phi, global float* sm_phi, float beta, int width, int height, int depth){

    int globId = (int)get_global_id(0);// From 0 to height*depth
    int localId = get_local_id(0);
    int grp_size = get_local_size(0);

    __local float loc_array[1840];//This hard coded size is (num_threads+ghosts)*(block_size+ghosts)*3 = 
    // 34*18*3 = 1836
    // 34*34*3 = 3468

    int curr = globId*width;//Current value 0, width, 2*width, .. -> init 0 row, init 1 row, init 2 row
    int slice = width*height;//This is the size of one 'slice'

    int globLocIdx =curr - slice - width - 1;//First index we need to access
    int locIdx= 0;
    int BLOCK_SIZE = 16;
    int ghosts = 2;
    int currBlock = 0;

    int row = width;//This is the size of one 'row'
    //(test if is last row)
    bool isLastRow = ( (curr+row) % slice == 0) ? true : false;
    //(test if is first row)
    bool isFirstRow =  (curr % slice == 0)? true : false;

    int locslice = (BLOCK_SIZE+2)*(grp_size + 2);
    int locrow = BLOCK_SIZE + ghosts;

    int totIter = (width/BLOCK_SIZE);

    //Defines the index to be incremented in case we are in the last or first row
    int dn = isLastRow? 0: locrow; //down value 
    int up = isFirstRow? 0: -locrow; //up value 

    // ----------------- If we are are in the last slice then we can't have a far slice
    int fcurr = locslice;//(far current) 
    // Test that we are not in the last slice
    if( curr > (slice*(depth-1) - 1) ){ fcurr = 0;}

    // ----------------- If we are are in the first slice then we can't have a closer slice
    // Closer 9 neighbors
    int ccurr = -locslice;//(closer current) 
    if( curr < slice ){ ccurr = 0;}

    int req_add_to_loc = localId*2;

    float16 a,b,c,d,e,f, dD;

    while(currBlock < totIter){
        locIdx= 0;

        //-------------------- Reads the required memory into local memory ----
        if(localId == 0){
            globLocIdx = curr - slice - width - 1 + currBlock*BLOCK_SIZE;//First index we need to access
            //This loop fills each of the 3 slices
            for(int fillSlice = 0; fillSlice < 3; fillSlice++){
                //This loop is for each of the rows
                for(int fillrow = 0; fillrow < grp_size + ghosts; fillrow++){
                    //Iterate over the columns
                    for(int col = 0; col < BLOCK_SIZE + ghosts; col++){
                        loc_array[locIdx] = phi[globLocIdx];
                        //loc_array[locIdx] = globLocIdx;
                        locIdx++;
                        globLocIdx++;
                    }
                    globLocIdx+= width - BLOCK_SIZE - ghosts;//Go to next row
                }
                //Position index in the next slice
                globLocIdx = curr - width +  fillSlice*slice - 1 + currBlock*BLOCK_SIZE;
            }
        }

        __local float* loc_phi  = loc_array + locslice + locrow + 1;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Read all the data
        // Same slice values 
        float16 currvec = vload16(localId,loc_phi+req_add_to_loc);//Current
        float16 lfvec = vload16(localId,loc_phi+req_add_to_loc-1);//left
        float16 rivec = vload16(localId,loc_phi+req_add_to_loc+1);//right
        float16 dnvec = vload16(localId,loc_phi+req_add_to_loc+dn); //down value 
        float16 upvec = vload16(localId,loc_phi+req_add_to_loc+up);//up value 
        //Far values 
        float16 fcurrvec = vload16(localId,loc_phi+req_add_to_loc+fcurr);//(far current) 
        // Closer values
        float16 ccurrvec = vload16(localId,loc_phi+req_add_to_loc+ccurr);//(closer current) 

        // Fix the boundaries
        if(currBlock == 0){//Fix all the left values
            lfvec.s0 = lfvec.s1;
        }
        if(currBlock == (width/BLOCK_SIZE) -1){//Fix all the rigth values
            rivec.sF = rivec.sE;
        }

        a = currvec - lfvec; // Backward in x 
        b = rivec - currvec; // Forward in x 
        c = currvec - upvec; // Backward in y 
        d = dnvec -  currvec; //Forward in y 
        e = currvec - ccurrvec; // Backward in z 
        f = fcurrvec  - currvec; // Forward in z 

        dD = temp_dD_vec(a,b,c,d,e,f, currvec);

        float16 sm_phiVal = currvec - beta * (currvec/sqrt( pow(currvec,2) + 1)) * dD;

        vstore16(sm_phiVal, (curr/BLOCK_SIZE)+currBlock, sm_phi);

        currBlock++;
    }//Iterates over the blocks

}

__kernel
void newphi( __global float* phi, __global float* dphidt,
        __global float* max_dphidt, int width, int height){

    float dt = .45/(max_dphidt[0] + EPS);
    int globId = (int)get_global_id(0);// From 0 to height*depth
    int curr = globId*width;//Current value

    int totIter = (width/16);
    int step = globId * totIter;

    float16  oldPhi;
    float16  dphidtVec;
    float16  newphi;

    //Iterate over the 'middle' columns
    for(int col = 0; col < totIter; col++){
        oldPhi= vload16(step + col,phi);//left
        dphidtVec= vload16(step + col,dphidt);//left
        //phi[curr+col] = phi[curr+col] + dt*dphidt[curr+col];
        vstore16(oldPhi + dt*dphidtVec, step + col, phi);
    }
}

__kernel
void reduce(__global float* buffer,
        __local float* scratch,
        __global float* result,
        __const int length,
        __const int absVal) {

    int global_index = get_global_id(0);
    float maxVal = 0;
    // Loop sequentially over chunks of input vector
    if( absVal){
        while (global_index < length) {
            float element = fabs(buffer[global_index]);
            maxVal = (maxVal > element) ? maxVal : element;
            global_index += get_global_size(0);
        }
    }else{
        while (global_index < length) {
            float element = buffer[global_index];
            maxVal = (maxVal > element) ? maxVal : element;
            global_index += get_global_size(0);
        }
    }

    // Perform parallel reduction
    int local_index = get_local_id(0);
    scratch[local_index] = maxVal;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int offset = get_local_size(0) / 2; offset > 0; offset = offset / 2) {
        if (local_index < offset) {
            float other = scratch[local_index + offset];
            float mine = scratch[local_index];
            scratch[local_index] = (mine > other) ? mine : other;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_index == 0) {
        result[get_group_id(0)] = scratch[0];
    }
}


__kernel void
dphidt(__global float* curvature, __global float* F,
        __global float* max_F, __global float* dphidt, float alpha, int width, int height){

    int globId = (int)get_global_id(0);// From 0 to height*depth
    int curr = globId*width;//Current value

    float maxF = max_F[0];//Max value of F

    //Iterate over the 'middle' columns
    float16  fVal;
    float16  dphidtVal;
    float16  curvVal;
    int totIter = (width/16);
    int step = globId * totIter;

    for(int col = 0; col < totIter; col++){
        fVal = vload16( step + col,F );
        curvVal = vload16( step + col,curvature);
        dphidtVal = (fVal/(maxF + EPS)) + alpha * curvVal;
        vstore16(dphidtVal, step + col, dphidt);
    }

}

/**
 * Computes the energy force of the Active Contour. It is divided one
 * thread per 'row', so there height*depth total threads. 
 */
__kernel void
compF(global float* avg_in_out, global float* I, 
        global float* F, int width, int height, int depth){

    int globId = (int)get_global_id(0);// From 0 to height*depth
    // This is the case we are in the middle of the cube, no worries about the
    // boundaries
    int curr = globId*width;//Current value

    // ----------------- If we are are in the last slice then we can't have a far slice
    float v = avg_in_out[0];// u is interior avg
    float u = avg_in_out[1];// v is exterior avg

    //Create temporal variables for faster response
    float16  currValI;
    float16  currValU;
    float16  currValV;

    int totIter = (width/16);
    int step = globId * totIter;

    float lamda1 = 1.1;
    float lamda2 = .9;
    for(int col = 0; col < totIter; col++){
        currValI = vload16(step + col,I);//left
        currValU = currValI - u;
        currValV = currValI - v;
        vstore16(lamda1*(currValU*currValU) - lamda2*(currValV*currValV), step + col, F);
    }

}//compF



// It computes the curvature of the curve phi. Each thread is in charge of
// evaluating a 'row' of elements. There are a total of 'height*depth' threads
__kernel void
curvature(global float* phi, global float* curvature, 
        int width, int height, int depth){

    int globId = (int)get_global_id(0);// From 0 to height*depth
    int localId = get_local_id(0);
    int grp_size = get_local_size(0);

    __local float loc_array[1840];//This hard coded size is (num_threads+ghosts)*(block_size+ghosts)*3 = 
    // 34*18*3 = 1836
    // 34*34*3 = 3468

    int curr = globId*width;//Current value 0, width, 2*width, .. -> init 0 row, init 1 row, init 2 row
    int slice = width*height;//This is the size of one 'slice'

    int globLocIdx =curr - slice - width - 1;//First index we need to access
    int locIdx= 0;
    int BLOCK_SIZE = 16;
    int ghosts = 2;
    int currBlock = 0;

    int row = width;//This is the size of one 'row'
    //(test if is last row)
    bool isLastRow = ( (curr+row) % slice == 0) ? true : false;
    //(test if is first row)
    bool isFirstRow =  (curr % slice == 0)? true : false;

    int locslice = (BLOCK_SIZE+2)*(grp_size + 2);
    int locrow = BLOCK_SIZE + ghosts;

    //int totIter = (width/BLOCK_SIZE);
    int totIter = (width/BLOCK_SIZE);

    //Defines the index to be incremented in case we are in the last or first row
    int dn = isLastRow? 0: locrow; //down value 
    int up = isFirstRow? 0: -locrow; //up value 

    // ----------------- If we are are in the last slice then we can't have a far slice
    int fcurr = locslice;//(far current) 
    // Test that we are not in the last slice
    if( curr > (slice*(depth-1) - 1) ){ fcurr = 0;}
    int fdn = isLastRow? fcurr : fcurr+locrow; //far down value 
    int fup = isFirstRow? fcurr : fcurr-locrow;//far up value

    // ----------------- If we are are in the first slice then we can't have a closer slice
    // Closer 9 neighbors
    int ccurr = -locslice;//(closer current) 
    if( curr < slice ){ ccurr = 0;}
    int cdn = isLastRow? ccurr : ccurr+locrow; //close down value 
    int cup = isFirstRow? ccurr : ccurr-locrow; //Close up value

    int req_add_to_loc = localId*2;

    while(currBlock < totIter){
        locIdx= 0;

        //-------------------- Reads the required memory into local memory ----
        if(localId == 0){
            globLocIdx = curr - slice - width - 1 + currBlock*BLOCK_SIZE;//First index we need to access
            //This loop fills each of the 3 slices
            for(int fillSlice = 0; fillSlice < 3; fillSlice++){
                //This loop is for each of the rows
                for(int fillrow = 0; fillrow < grp_size + ghosts; fillrow++){
                    //Iterate over the columns
                    for(int col = 0; col < BLOCK_SIZE + ghosts; col++){
                        loc_array[locIdx] = phi[globLocIdx];
                        //loc_array[locIdx] = globLocIdx;
                        locIdx++;
                        globLocIdx++;
                    }
                    globLocIdx+= width - BLOCK_SIZE - ghosts;//Go to next row
                }
                //Position index in the next slice
                globLocIdx = curr - width +  fillSlice*slice - 1 + currBlock*BLOCK_SIZE;
            }
        }

        __local float* loc_phi  = loc_array + locslice + locrow + 1;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Read all the data
        // Same slice values 
        float16 currvec= vload16(localId,loc_phi+req_add_to_loc);//left
        float16 lfvec = vload16(localId,loc_phi+req_add_to_loc-1);//left
        float16 rivec = vload16(localId,loc_phi+req_add_to_loc+1);//right
        float16 dnvec = vload16(localId,loc_phi+req_add_to_loc+dn); //down value 
        float16 drvec = vload16(localId,loc_phi+req_add_to_loc+dn+1);// (down right)
        float16 dlvec = vload16(localId,loc_phi+req_add_to_loc+dn-1);// (down left)
        float16 upvec = vload16(localId,loc_phi+req_add_to_loc+up);//up value 
        float16 urvec = vload16(localId,loc_phi+req_add_to_loc+up+1);// (up right)
        float16 ulvec = vload16(localId,loc_phi+req_add_to_loc+up-1);// (up left)
        //Far values 
        float16 fcurrvec = vload16(localId,loc_phi+req_add_to_loc+fcurr);//(far current) 
        float16 flfvec = vload16(localId,loc_phi+req_add_to_loc+fcurr-1);// (far left)
        float16 frivec = vload16(localId,loc_phi+req_add_to_loc+fcurr+1);//(far right) 
        float16 fdnvec = vload16(localId,loc_phi+req_add_to_loc+fdn); //far down value 
        float16 fdrvec = vload16(localId,loc_phi+req_add_to_loc+fdn+1);// (far down right)
        float16 fdlvec = vload16(localId,loc_phi+req_add_to_loc+fdn-1);// (far down left)
        float16 fupvec = vload16(localId,loc_phi+req_add_to_loc+fup);//far up value
        float16 furvec = vload16(localId,loc_phi+req_add_to_loc+fup+1);// (up right)
        float16 fulvec = vload16(localId,loc_phi+req_add_to_loc+fup-1);// (up left)
        // Closer values
        float16 ccurrvec = vload16(localId,loc_phi+req_add_to_loc+ccurr);//(closer current) 
        float16 clfvec =vload16(localId,loc_phi+req_add_to_loc+ccurr-1);// (closer left)
        float16 crivec =vload16(localId,loc_phi+req_add_to_loc+ccurr+1);//(closer right) 
        float16 cdnvec =vload16(localId,loc_phi+req_add_to_loc+cdn); //close down value 
        float16 cdrvec =vload16(localId,loc_phi+req_add_to_loc+cdn+1);// (closer down right)
        float16 cdlvec =vload16(localId,loc_phi+req_add_to_loc+cdn-1);// (closer down left)
        float16 cupvec =vload16(localId,loc_phi+req_add_to_loc+cup); //Close up value
        float16 curvec =vload16(localId,loc_phi+req_add_to_loc+cup+1);// (closer up right)
        float16 culvec =vload16(localId,loc_phi+req_add_to_loc+cup-1);// (closer up left)

        // Fix the boundaries
        if(currBlock == 0){//Fix all the left values
            lfvec.s0 = lfvec.s1;
            dlvec.s0 = dlvec.s1;
            ulvec.s0 = ulvec.s1;
            flfvec.s0 = flfvec.s1;
            fdlvec.s0 = fdlvec.s1;
            fulvec.s0 = fulvec.s1;
            clfvec.s0 = clfvec.s1;
            cdlvec.s0 = cdlvec.s1;
            culvec.s0 = culvec.s1;
        }
        if(currBlock == (width/BLOCK_SIZE) -1){//Fix all the rigth values
            rivec.sF = rivec.sE;
            drvec.sF = drvec.sE;
            urvec.sF = urvec.sE;
            frivec.sF = frivec.sE;
            fdrvec.sF = fdrvec.sE;
            furvec.sF = furvec.sE;
            crivec.sF = crivec.sE;
            cdrvec.sF = cdrvec.sE;
            curvec.sF = curvec.sE;
        }

        // Assuming h = .5
        // Compute finite differences
        //First order
        float16 phi_16_x = (rivec - lfvec);
        float16 phi_16_y = (upvec - dnvec);
        float16 phi_16_z = (fcurrvec - ccurrvec);
        //Second order
        float16 phi_16_xx = (lfvec - 2*currvec + rivec);
        float16 phi_16_yy = (dnvec - 2*currvec + upvec);
        float16 phi_16_zz = (ccurrvec - 2*currvec + fcurrvec);

        float16 phi_16_xy = 0.25f*(drvec - ulvec + urvec - dlvec );
        float16 phi_16_xz = 0.25f*(frivec - clfvec + flfvec - crivec);
        float16 phi_16_zy = 0.25f*(fupvec - cdnvec + fdnvec - cupvec);
        //Squares
        float16 phi_16_x2 = phi_16_x*phi_16_x;
        float16 phi_16_y2 = phi_16_y*phi_16_y;
        float16 phi_16_z2 = phi_16_z*phi_16_z;

        float16 newcurv  =   ( phi_16_x2*phi_16_yy + phi_16_x2*phi_16_zz + phi_16_y2*phi_16_xx + 
                phi_16_y2*phi_16_zz + phi_16_z2*phi_16_xx + phi_16_z2*phi_16_yy
                - 2*phi_16_x*phi_16_y*phi_16_xy -2*phi_16_x*phi_16_z*phi_16_xz - 2*phi_16_y*phi_16_z*phi_16_zy) / 
            pow((phi_16_x2 + phi_16_y2 + phi_16_z2 + EPS),(float)(1.5));//It doesn't work if use 3/2 it does ceil or top

        vstore16(newcurv, (curr/BLOCK_SIZE)+currBlock, curvature);

        currBlock++;
    }//Iterates over the blocks

}//curvature

int indxFromCoordAC(int width, int height, int row, int col, int dim){
    return width*row*dim + dim*col;
}

int indxFromCoord3D(int width, int height, int depth,
        int row, int col, int z, int dim){
    return width*height*z + width*row + col;
}

/**
 * Copies a 3D image into a buffer. 
 * The image should only contain one band.
 */
__kernel void textToBufNew(write_only image3d_t in, global float* buf){

    int width = get_image_width(in);
    int height = get_image_height(in);
    int depth = get_image_depth(in);

    int w_by_h = width*height;
    int idx = (int)get_global_id(0);

    int z = (int)(idx/w_by_h);
    int prevCube  = (z-1)*w_by_h;

    int row = (int)((idx - prevCube)/width);
    int col = idx - prevCube - (row-1)*width ;

    //write_imagef(in,def_sampler, (int4)(col,row,z,1));
}

__kernel void textToBuf(read_only image3d_t in, global float* buf){
    int width = get_image_width(in);
    int height = get_image_height(in);
    int depth = get_image_depth(in);

    int oneDidx= (int)get_global_id(0);

    /*
       int size = width*height*depth;
       int z = ceil(oneDidx/(width*height));//Which depth are we
       int col = (int) 
       int row = 

       float4 textVal = read_imagef(in, def_sampler, (int4)(col,row,z,1));

       buf[oneDidx] = textVal.x; 
       */
}

/**
 * This kernel copies one buffer into an image. 
 * If 'allBands' is true, then the buffer should contain the 4 bands on it.
 * If 'allBands' is false, then the buffer should contain only information
 * in one channel, and it is copied into the 3 channels RGB of the image
 */
__kernel void bufToText(global float* buf, write_only image2d_t out, 
        int width, int height, int allBands){

    int col = (int)get_global_id(0);
    int row = (int)get_global_id(1);

    int currIndx = indxFromCoordAC(width, height, row, col, 1);

    float4 textVal;
    if(allBands){
        float red = buf[currIndx*4];
        float green = buf[currIndx*4 + 1];
        float blue = buf[currIndx*4 + 2];
        float alpha = buf[currIndx*4 + 3];

        textVal = (float4)(red, green, blue, alpha);
    }else{
        float val = buf[currIndx];
        textVal = (float4)(val, val, val, 1);
    }

    write_imagef(out, (int2)(col,row), textVal);
}

/**
 * This kernel computes local averages of pixels inside and outside the object
 * for every warp size. It only works for positive values 
 * How we are doing it is as follows:
 * Each group of will process (LocalMemSize / width) lines each time.
 * @param width
 * @param height 
 * @param depth
 * @param width
 */
__kernel void
avgInOut(const __global float* phi,const  __global float* img_in,
        __global float* avg_in_out, __const int size, 
        __const int cellsPerWorkItem) {

    int indx = (int)get_global_id(0)*cellsPerWorkItem;//Start index for this work item
    int origIndx = indx;//Save the initial index
    int totalWorkItems = (int)get_local_size(0);//Items in this group

    float value = 0;

    //This are local variables to compute the final reduction
    local int currCountOutAll;
    local int currCountInAll;
    local int currSumOutAll;
    local int currSumInAll;

    // Only first thread initializes these local variables
    if( get_local_id(0) == 0){
        currCountOutAll= 0;
        currCountInAll = 0;
        currSumOutAll = 0;
        currSumInAll = 0;
    }

    //This are private variables to compute reduction
    float currSumIn = 0;
    float currSumOut = 0;
    int currCountIn = 0;
    int currCountOut = 0;
    int iter = 0;

    while(indx < size){//Indicates when each thread should stop
        //This is the number of cells that are computed for each thread
        for(int i = 0; i < cellsPerWorkItem; i++){
            if(indx < size){
                value = img_in[indx];
                if(phi[indx] > 0){
                    // Pixels outside the mask
                    currSumOut = currSumOut + value;
                    // Count pixels outside the mask
                    currCountOut = currCountOut + 1;
                }else{
                    // Pixels inside the mask
                    currSumIn = currSumIn + value;
                    // Count pixels inside the mask
                    currCountIn = currCountIn + 1;
                }
            }else{//We already finish
                break;
            }
            indx ++;//We increment the current index
        }//For

        // Assuming that we are using the right amount of memory is better if wersynchronize here

        iter++;// Increment the iteration we are computing
        indx = (int)origIndx + iter*cellsPerWorkItem*totalWorkItems;
    }

    //Adding atomically to the output variable
    atomic_add(&currCountOutAll, currCountOut);
    atomic_add(&currCountInAll, currCountIn);
    atomic_add(&currSumOutAll, (int)currSumOut);
    atomic_add(&currSumInAll,(int)currSumIn);

    barrier(CLK_LOCAL_MEM_FENCE);

    // Only thread 1 makes the final computation
    if( get_local_id(0) == 0){
        avg_in_out[0] = (float)currSumOutAll/((float)currCountOutAll+EPS);
        avg_in_out[1] = (float)(currSumInAll/((float)currCountInAll+EPS));
        avg_in_out[2] = (float)currCountOutAll;
        avg_in_out[3] = (float)currCountInAll;
        avg_in_out[4] = (float)currSumOutAll;
        avg_in_out[5] = (float)currSumInAll;
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
