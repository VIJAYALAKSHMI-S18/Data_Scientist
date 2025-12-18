import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib
df=pd.read_csv(r'C:\Users\svija\Documents\Data_Scientist\KNN ALGORITHM PROJECT\Social_Network_Ads.csv')
x=df.drop('Purchased',axis=1)
y=df['Purchased']
AI_model=KNeighborsClassifier(n_neighbors=300)
AI_model.fit(x,y)
# new_data1=int(input("ENTER AGE:"))
# new_data2=int(input("ENTER SALARY:"))
# p=AI_model.predict([[new_data1,new_data2]])
joblib.dump(AI_model,'Social_Network_Streamlit.pkl')
print("Model Saved Successfully!!")