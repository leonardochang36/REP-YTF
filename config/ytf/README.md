## The rep_ytf_config.mat file contains the following MATLAB variables.

-videoList:
	3425x1 cell array containing all the 3425 face video filenames of the YTF database. Note that only filenames are contained, without directories or full paths. Note also that the following variables are all based on the order of this video list. Therefore the extracted features must exactly follow the order of this list.
	
-imgGalleryList:
	3425x1 cell array containing all the best 3425 face images filenames of each video of the YTF database, that compase the image gallery. Note that only filenames are contained, without directories or full paths. Therefore the extracted features must exactly follow the order of this list in order to form the image gallery.

-labels:
	3425x1 vector containing class labels associated with the above video list. There are 1595 classes and the labels range from 1 to 1595.

-trainIndex:
	10x1 cell array, with each cell containing the indexes of the training videos for one trial.

-testIndex:
	10x1 cell array, with each cell containing the indexes of the test images for one trial.

-galIndex:
	10x3 cell array, with each cell containing the indexes of the gallery videos for each openness value for one trial. For each trial t, and openness op, galIndex{t,op} is a subset of testIndex{t}.

-probIndex:
	10x3 cell array, with each cell containing the indexes of the probe videos for each openness value for one trial. For each trial t, and openness op, probIndex{t,op} is a subset of testIndex{t}. 
