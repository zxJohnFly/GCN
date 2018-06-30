#include <stdio.h>

double mAP(int *cateTrainTest, int *IX, int topk, int m, int n, double *mAPs){
    int i, j, idx;
    double x, p, mAP;

    for (i = 0; i < n; i++) {
        x = 0.0;
        p = 0.0;

        for (j = 0; j < topk; j++) {
           idx = IX[i + n*j];  
           if (cateTrainTest[i + n*idx] == 1){
               x += 1.0;
               p = p + x/(j*1.0 + 1.0);
           }
        }

        if (p == 0.0){
            mAPs[i] = 0.0;
        }
        else{
            mAPs[i] = p/x;
        }
    } 

    mAP = 0.0;
    for (i = 0; i < n; i++){
        mAP += mAPs[i];
    }
    mAP = mAP/n;

    return mAP;
}

void topK(int *cateTrainTest, int *IX, int topk, int m, int n, double *precs, double *recs){
    int i, j, idx;
    int retrieved_rel, real_rel;

    for (i = 0; i < n; i++){
        retrieved_rel = 0;
        for (j = 0; j < topk; j++){
            idx = IX[i + n*j];
            retrieved_rel += cateTrainTest[i + n*idx];
        }
        
        real_rel = 0;
        for (j = 0; j < m; j++) real_rel += cateTrainTest[i + n*j];
        
        precs[i] = retrieved_rel/(topk*1.0);
        recs[i] = retrieved_rel/(real_rel*1.0);
    }
}


