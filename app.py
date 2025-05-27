from flask import Flask, render_template, request
import pandas as pd
import joblib
import networkx as nx

app = Flask(__name__)

# Load airport graph
edges_df = pd.read_csv('data/airport_edges_delhi.csv')
nodes_df = pd.read_csv('data/labeled_airport_nodes_delhi.csv')
label_map = pd.read_csv('data/node_labels.csv')
label_map = {int(row['osmid']): row['readable_label'] for _, row in label_map.iterrows()}

G = nx.DiGraph()
for _, row in edges_df.iterrows():
    G.add_edge(row['u'], row['v'], weight=row['length'])

# Load ML model
model = joblib.load('delay_model.pkl')
weather_encoder = joblib.load('weather_encoder.pkl')

def predict_delay(time_of_day, traffic_level, weather):
    w = weather_encoder.transform([weather])[0]
    return model.predict([[time_of_day, traffic_level, w]])[0]

def dijkstra_path(start, end):
    try:
        path = nx.dijkstra_path(G, start, end, weight='weight')
        time = nx.dijkstra_path_length(G, start, end, weight='weight')
        return path, round(time, 2)
    except nx.NetworkXNoPath:
        return [], "No Path"

@app.route('/', methods=['GET', 'POST'])
def index():
    nodes = sorted(G.nodes)
    edges = [{'from': u, 'to': v, 'label': str(G[u][v]['weight'])} for u, v in G.edges]
    result = None

    if request.method == 'POST':
        start = int(request.form['start'])
        end = int(request.form['end'])
        time_of_day = int(request.form['time_of_day'])
        traffic_level = int(request.form['traffic_level'])
        weather = request.form['weather']

        path, raw_time = dijkstra_path(start, end)
        
        if raw_time != "No Path":
            travel_time = round(float(raw_time) / 15, 2)
            raw_distance = round(float(raw_time), 2)
            delay = predict_delay(time_of_day, traffic_level, weather)
            total_estimate = round(travel_time + delay, 2)
        else:
            travel_time = "No Path"
            raw_distance = "No Path"
            delay = None
            total_estimate = None
        readable_path = [label_map.get(n, str(n)) for n in path]
        result = {
            'path': path,
            'readable_path': readable_path,
            'time': travel_time,
            'predicted_delay': round(delay, 2) if delay is not None else "N/A",
            'raw_distance': raw_distance,
            'total_estimate': total_estimate if total_estimate is not None else "N/A"
        }

        
        if isinstance(travel_time, (float, int)):
            result['total_estimate'] = round(travel_time + delay, 2)



    return render_template('visual.html', nodes=nodes, edges=edges, result=result, labels=label_map)

if __name__ == '__main__':
    app.run(debug=True)
