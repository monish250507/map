# GreenRoute AI - Eco Routing System

An intelligent eco-routing API that optimizes vehicle routes for fuel efficiency, reduced emissions, and sustainability using machine learning predictions.

## Features

- **Eco-Friendly Route Optimization**: Find routes that minimize fuel consumption and CO2 emissions
- **ML-Powered Predictions**: Traffic forecasting, fuel consumption estimation, driver profiling, and route risk detection
- **Multiple Optimization Preferences**: Choose between `green`, `fast`, or `balanced` routing strategies
- **Comprehensive Metrics**: Get detailed breakdowns including fuel consumption, CO2 emissions, eco scores, and segment-by-segment analysis
- **Explainable AI**: Understand why routes were chosen with detailed explanations
- **RESTful API**: FastAPI-based backend with OpenAPI documentation

## Architecture

```
eco_routing/
├── api/              # API routes, authentication, health checks
├── config/           # Configuration and API key management
├── core/             # Core routing logic and agents
│   ├── cost_function.py      # Segment cost calculation
│   ├── eco_score.py          # Eco score computation
│   ├── explorer_agent.py     # Main orchestrator
│   ├── fuel_emission_agent.py # Fuel and emission calculations
│   ├── pathfinder.py         # Route pathfinding algorithms
│   └── road_graph.py         # OSM road graph management
├── data/             # Data loaders and preprocessing
├── ml/               # Machine learning models
│   ├── driver_profiler.py        # Driver behavior profiling
│   ├── fuel_predictor.py         # Fuel consumption prediction
│   ├── route_risk_detector.py    # Route risk assessment
│   └── traffic_predictor.py      # Traffic prediction
└── models/           # Data models (driver, vehicle, environment)
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd map
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API keys (optional):
   - Add API keys to `eco_routing/config/api_keys.txt`
   - Or set environment variables with `ECOROUTING_` prefix

## Usage

### Starting the Server

```bash
python run_server.py
```

The server will start on `http://127.0.0.1:8000`

### API Endpoints

#### Health Check
```
GET /eco-route/health
```

#### Get Eco Route
```
POST /eco-route/eco-route
```

**Request Body:**
```json
{
  "source": {
    "lat": 19.0760,
    "lon": 72.8777
  },
  "destination": {
    "lat": 18.5204,
    "lon": 73.8567
  },
  "vehicle_id": "vehicle_123",
  "driver_id": "driver_456",
  "preference": "green"
}
```

**Response:**
```json
{
  "route_coordinates": [
    {"lat": 19.0760, "lng": 72.8777},
    ...
  ],
  "distance_km": 150.5,
  "time_min": 120,
  "fuel_consumption": 12.3,
  "co2_emissions": 28.9,
  "eco_score": 85.5,
  "explanation": "Route optimized for minimal fuel consumption",
  "ml_outputs": {
    "traffic_prediction": {...},
    "fuel_prediction": {...},
    "risk_assessment": {...}
  },
  "segment_breakdown": [...]
}
```

#### Optimize Route (Legacy)
```
POST /eco-route/optimize
```

Returns GeoJSON format with full route details.

### Authentication

All endpoints require API key authentication via the `X-API-Key` header:

```bash
curl -X POST http://127.0.0.1:8000/eco-route/eco-route \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"source": {"lat": 19.0760, "lon": 72.8777}, ...}'
```

## Configuration

Environment variables (prefixed with `ECOROUTING_`):

- `OSM_DOWNLOAD_RADIUS_M`: Radius for OSM graph download (default: 20000m)
- `MAX_BBOX_KM`: Maximum bounding box span (default: 120km)
- `BBOX_MARGIN_KM`: Bounding box margin (default: 20km)
- `GRAPH_CACHE_SIZE`: Number of graphs to cache (default: 4)
- `DATA_DIR`: Path to data files directory

## Machine Learning Models

The system uses several ML models:

- **Traffic Predictor**: Forecasts traffic conditions on route segments
- **Fuel Predictor**: Estimates fuel consumption based on vehicle, driver, and route characteristics
- **Driver Profiler**: Infers driver behavior patterns (eco vs. aggressive)
- **Route Risk Detector**: Assesses safety and risk factors along routes

## Route Preferences

- **`green`**: Prioritizes fuel efficiency and emissions reduction (40% weight on fuel, 30% on emissions)
- **`fast`**: Prioritizes travel time and speed (35% weight on time, 30% on speed)
- **`balanced`**: Balanced approach across all factors (25% weight on fuel, 25% on time)

## Development

### Project Structure

- `run_server.py`: Server entry point
- `base_price_prediction.py`: Base price prediction model (standalone)
- `eco_routing/`: Main application package

### Testing

```bash
# Run health check
curl http://127.0.0.1:8000/eco-route/health
```

### API Documentation

Once the server is running, visit:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Dependencies

- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `pydantic`: Data validation
- `scikit-learn`: Machine learning

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

