### Overview- Energy Generation a Global Perspective

This project uses data from the World Data Institute to predict Energy Generation classes (very low, low, mid, high) for 35000 power plants located around the world. This dataset utilizes various factors, like fuel type, capacity, energy generation data from 2013-2019, and location. These variables are important for determined which class the powerplant 

Deployment Link: https://energy-model-deploy-wusapp6xmuzqhvbfeumta6n.streamlit.app/
Deployment Github: https://github.com/silver032/Energy-Model-Deploy
Tableau Public Link: 
https://public.tableau.com/views/globalpowerplantwbvis1/GenerationClasses?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link
https://public.tableau.com/views/globalpowerplantwbvis2/CapacityGenerationovertime?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link
https://public.tableau.com/views/globalpowerplantwbvis3/GlobalDistribution?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link
https://public.tableau.com/views/globalpowerplantwbvis4/TotalC02Emissions?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link
https://public.tableau.com/views/globalpowerplantwbvis5/GenerationvsCapcitybyFueltype?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link

# Dataset Description

The dataset includes several variables pertaining the the plants. Capcaity, generation data, location, year built, data source were all included. 

# Business Problem 

The project is comissioned by the Internation Panel of Climate Change to gain a better understanding of global energy generation. This data will be used to make policy decisions as we transition to a greener future. 

What impact will it have if we can perdict the generation class of power plants?
What are some of the leading features that lead to this classification?
What is the global distribution of power plants and type, and how can this data be used for future policy decisions?

# Goals

Predictive Analysis: To predict generation class (very low, low, mid, high) of different power plants across the world.. 
Indicator Analysis: To identify key features that contirbute to this classification. 

# Key Questions

1. What preprocessing steps do we need to take to create an effective predictive model?
2. Which type of predictive model gets the best results?
3. What hypertuning techniques are used to tune the model?
4. What are the benefits of our model?

# Target Variable Investigation

In the target variable, average generation, we started with four classes:
very low
low 
mid
high

And tested different models to determine the best results. 
The notebook contains the following types of content:

## Preprocessing Steps...
Converted commission year to plant age
Removed plants with multiple fuel types because distribution is not known, focsuing on the majority
Grouping renewable fuels
Calculating Average Generation from actual, using estimated if not present
Inpute missing plant ages by mean
Recalculate average generation and binning into generation classes

## Exploratory data analysis
# Explored Descriptive statistics
![finalimage1](https://github.com/silver032/Energy-Generation-Classifier-phase-5-project-/blob/main/images/finalimage1.png)

# Capacity by Fuel type
![finalimage2](https://github.com/silver032/Energy-Generation-Classifier-phase-5-project-/blob/main/images/finalimage2.png)

# Total Generation by Fuel type
![finalimage3](https://github.com/silver032/Energy-Generation-Classifier-phase-5-project-/blob/main/images/finalimage3.png)

# Geographical distribution of Plants 
![finalimage4](https://github.com/silver032/Energy-Generation-Classifier-phase-5-project-/blob/main/images/finalimage4.png)

# Capacity by fuel time and Generation
![finalimage5](https://github.com/silver032/Energy-Generation-Classifier-phase-5-project-/blob/main/images/finalimage5.png)

# Class distribution
![finalimage6](https://github.com/silver032/Energy-Generation-Classifier-phase-5-project-/blob/main/images/finalimage6.png)

#Class distribution by Fuel type
![finalimage7](https://github.com/silver032/Energy-Generation-Classifier-phase-5-project-/blob/main/images/finalimage7.png)

## Model Results
# Baseline Model Results
![finalimage8](https://github.com/silver032/Energy-Generation-Classifier-phase-5-project-/blob/main/images/finalimage9.png)

# Hyperparameter Model results with Randomizedcvsearch and data augmention
![finalimage11](https://github.com/silver032/Energy-Generation-Classifier-phase-5-project-/blob/main/images/finalimage11.png)

# Confusion matrix for best tuned model
![finalimage13](https://github.com/silver032/Energy-Generation-Classifier-phase-5-project-/blob/main/images/finalimage13.png)

# Feature Importance for best model
![finalimage12](https://github.com/silver032/Energy-Generation-Classifier-phase-5-project-/blob/main/images/finalimage12.png)


## Conclusions, next steps...
1. Effectiveness of the Random Forest Classifier: 94.6 - Highly Effective
2. Significance of Feature Importance: Capacity and Primary fuel 
3. The Value of Data Augmentation and Scalability: more robust model 
4. Streamlit Deployment as an Accessible Solution and to gather more data
5. Insights for Policy and Operational Efficiency: Target High generation dirty fuels
6. Broader Implications for Energy Management: Grid optimization for developing countries
Summary
This project highlighted the potential of machine learning in predicting power generation classes across global power plants. The approach proved to be effective in tackling real-world energy generation challenges, offering insights into improving efficiency and optimizing resource management across the global power sector.
Next Steps
Model Refinement - more data, other factors like environmental factors
Dashboard Enhancements:
Explore Predictive Maintenance: Operational failures and maintenance 
Integration with Real-World Systems:
Ongoing Deployment and Monitoring: Utilizing input data from streamlit app and other sources

