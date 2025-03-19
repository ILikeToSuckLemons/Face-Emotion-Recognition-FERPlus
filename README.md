I made this project to accurately predict people's facial emotion. I yielded over 80% testing accuracy, please follow the repo step by step in order to get similar results

I used the Dataset of the FER2013 and FER2013plus for a better accuracy result.
If you want to yield similar results to this project, I suggest to download the 2 datasets before continuing.

FER2013.csv = this file contains only 7 emotions which are happiness, sadness, surprise, anger, disgust, fear and neutral
FER2013plus = this file has an additional emotions called "contempt". On top of that, the emotion scores for each image are distributed. Meaning that there will not be a definite emotion.

For Example, this is FER2013 labels:


![Alt Text](images/fer.png)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

While this is the FER2013plus labels:


![My Image](images/ferplus.jpg)


I will be using pytorch and cnn neural network model since it is best for reading images.

![My Image](images/FlowChart.jpg)


