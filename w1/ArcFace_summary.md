# ArcFace: Additive Angular Margin Loss for Deep Face
## Paper Date 9-Febuary-2019
## Dataset
LFW contant more than 13,233 images of faces collected from web. Each face has been labeled with the name of the person. 1680 of the people pictured have two or more distinct photos in the data set.
## Method
![alt text] (https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.weak-learner.com%2Fblog%2F2020%2F08%2F29%2FArcFace-loss%2F&psig=AOvVaw0LYk9zxlQps8IeuWod6zbX&ust=1612409073452000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCJCMuoPizO4CFQAAAAAdAAAAABAD)
1. from xi and W normalisation, get cos(theta_j).
2. Caculate arccos(theta_yi) and get the ange between the feature x_i and the ground truth W_yi
3. Add an angular margin penalty m on the target (ground truth) angle theta_yi
4. Caculate cos(theta_yi+m) and multiply all logits by the feature scale s
## Novelty
Use additive cosine margin insteed of multiplicative angular margin (SphereFace) and additive cosine margin (CosFace)
## Result
verification results
with margin=0.5, ArcFace get 99.53 on LFW, 95.56 on CFP-PF, 95.15 on AgeDB-30