# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the saved model

filename = 'hotel-occupancy-prediction-model.pkl'
regressor = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    trading_area_mapping = {'Airports Area': 0,'Bristol/Somerset': 1,'Central London Area': 2,'Cheshire/Shropshire': 3,'Cornwall/Devon': 4,\
		'East Anglia': 5,'East Midlands': 6,'Essex': 7,'Germany Northwest': 8,'Germany Southeast': 9,\
		'Gloucester/Worcester/Hereford': 10,'Greater Manchester': 11,'Inner London Area': 12,'Kent': 13,\
		'Lancashire/Cumbria': 14,'Leeds/Bradford': 15,'M4 Corridor': 16,'Merseyside': 17,'North East': 18,\
		'North Home Counties': 19,'Northern Ireland': 20,'Outer London Area': 21,'Oxfordshire/Buckinghamshire': 22,\
		'Peak District/Lincolnshire': 23,'Portsmouth/Southampton': 24,'Republic of Ireland': 25,'Scotland East': 26,\
		'Scotland North': 27,'Scotland West': 28,'South Coast': 29,'Sussex/Surrey': 30,'Wales': 31,'West Midlands': 32,'Yorkshire': 33}
    
    temp_array = []
    
    if request.method == 'POST':
       
        total_vws = request.form.get('total_vws',0.0)
        family_rooms = request.form.get('family_rooms',0.0)
        total_rooms_sold = request.form.get('total_rooms_sold',0.0)
        avgnights = request.form.get('avgnights',0.0)
        totalchildren = request.form.get('totalchildren',0.0)
        totalgrossrevenue_room = request.form.get('totalgrossrevenue_room',0.0)
        totalnetrevenue_breakfast = request.form.get('totalnetrevenue_breakfast',0.0)
        totalnetrevenue_mealdeal = request.form.get('totalnetrevenue_mealdeal',0.0)
        mobile_app = request.form.get('mobile_app',0.0)
        corporate_booking_tool = request.form.get('corporate_booking_tool',0.0)
        front_desk = request.form.get('front_desk',0.0)
        ccc = request.form.get('ccc',0.0)
        travelport_gds = request.form.get('travelport_gds',0.0)
        agency = request.form.get('agency',0.0)
        germany_web_de = request.form.get('germany_web_de',0.0)
        amadeus_gds = request.form.get('amadeus_gds',0.0)
        hub_mobile_app = request.form.get('hub_mobile_app',0.0)
        booking_com = request.form.get('booking_com',0.0)
        Canxrooms_last7days = request.form.get('Canxrooms_last7days',0.0)
        off_rooms = request.form.get('off_rooms',0.0)
        flex_rate = request.form.get('flex_rate',0.0)
        avg_revenue_per_room = request.form.get('avg_revenue_per_room',0.0)
        
        temp_array.extend([total_vws,family_rooms,total_rooms_sold,avgnights,totalchildren,totalgrossrevenue_room,\
                          totalnetrevenue_breakfast,totalnetrevenue_mealdeal,mobile_app,corporate_booking_tool,\
                          front_desk,ccc,travelport_gds,agency,germany_web_de,amadeus_gds,hub_mobile_app,booking_com,\
                          Canxrooms_last7days,off_rooms,flex_rate,avg_revenue_per_room]) 
                
        air_conditioned_rooms = request.form['air-conditioned-rooms']
        if air_conditioned_rooms == 'No':
            temp_array.extend([0])
        elif air_conditioned_rooms == 'Yes':
            temp_array.extend([1])
        
        london_region_split = request.form['london-region-split']
        if london_region_split == 'Regions':
            temp_array.extend([0,0])
        elif london_region_split == 'Germany':
            temp_array.extend([1,0])
        elif london_region_split == 'London':
            temp_array.extend([0,1])
           
        trading_area = request.form['trading-area']
        for key, value in trading_area_mapping.items():
             if trading_area == key:
                temp_array.append(value) 
                    
        # Convert the string to a float
        try:
            temp_array = [float(i) for i in temp_array]
        except ValueError:
            # Handle the case when the input is not a valid float
            return "Invalid input. Please enter a valid float value."
        
        data = np.array([temp_array])
        my_prediction = regressor.predict(data)[0]
              
        return render_template('result.html', lower_limit = round(my_prediction-2,2), upper_limit = round(my_prediction+2,2))


if __name__ == '__main__':
    #app.run(debug=True)
    # start server with 81 port
    app.run(debug=True)
