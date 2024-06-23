
import pandas as pd
import streamlit as st 
import joblib

# Load the trained model and encoders
model = joblib.load('model.pkl')
encoder_dict =joblib.load('encoder.pkl')
cols=['movement_reactions', 'mentality_composure', 'potential', 'wage_eur', 'value_eur', 'passing']    
  
def main(): 
    st.title("Players Rating Predictor")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Players Rating Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
    movement_reactions = st.number_input("Movement Reactions", min_value=0, max_value=100, value=50) 
    mentality_composure = st.number_input("Mentality Composure", min_value=0, max_value=100, value=50) 
    potential = st.number_input("Potential", min_value=0, max_value=100, value=50) 
    wage_eur = st.number_input("Wage (EUR)", min_value=0, value=10000) 
    value_eur = st.number_input("Value (EUR)", min_value=0, value=1000000) 
    passing = st.number_input("Passing", min_value=0, max_value=100, value=50) 
    
    if st.button("Predict"): 
        features = [[movement_reactions, mentality_composure, potential, wage_eur, value_eur, passing]]
        data = {
            'movement_reactions': movement_reactions, 
            'mentality_composure': mentality_composure, 
            'potential': potential, 
            'wage_eur': wage_eur, 
            'value_eur': value_eur, 
            'passing': passing
        }

        df = pd.DataFrame([data], columns=cols)
                
        # Ensure data is in the correct format for prediction
        for col in df.select_dtypes(include='object').columns:
            df[col] = encoder_dict[col].transform(df[col])

        features_list = df.values.tolist()      
        prediction = model.predict(features_list)
    
        output = int(prediction[0])
        
        st.success('Predicted Player Overall Rating is {}'.format(output))
      
if __name__ == '__main__': 
    main()