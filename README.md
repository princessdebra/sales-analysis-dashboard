# sales-analysis-dashboard
# Myntra Sales Analytics Dashboard

## Overview
A comprehensive sales analytics dashboard. This dashboard provides powerful insights into sales performance, customer behavior, and product trends using advanced data visualization and AI-powered analysis.

## Key Features

### 📊 Data Visualization
- Interactive charts and graphs for sales trends
- Category and brand performance analysis
- Geographical sales distribution
- Customer demographics visualization

### 🤖 AI-Powered Insights
- Automated sales trend analysis
- Performance benchmarking
- Predictive analytics
- Natural language query processing

### 🔍 Advanced Analytics
- Correlation analysis between metrics
- Low performance identification
- Optimization recommendations
- Seasonal trend detection

### 🔒 Secure Authentication
- Role-based access control
- Password hashing for security
- Session management

## Technologies Used

### Core Stack
- **Python** (Primary programming language)
- **Streamlit** (Web application framework)
- **Pandas** (Data manipulation)
- **Plotly** (Interactive visualizations)
- **Google Gemini API** (AI-powered analysis)

### Data Processing
- **NumPy** (Numerical operations)
- **Scikit-learn** (Machine learning components)
- **SciPy** (Statistical functions)

### Authentication
- **SHA-256** (Password hashing)
- **Session state management**

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd myntra-analytics-dashboard
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file
   - Add your Gemini API key:
     ```
     GEMINI_API_KEY=your_api_key_here
     ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

### Authentication
1. Register as a new user or login with existing credentials
2. Different roles available (admin, merchant, etc.)

### Dashboard Features
1. **Sales Overview** - Key metrics at a glance
2. **Trend Analysis** - Time-based sales patterns
3. **Category/Brand Performance** - Compare different segments
4. **Geographical Insights** - State and city-level data
5. **AI Assistant** - Ask natural language questions

### Data Interaction
- Filter by date range, state, and category
- Download filtered datasets
- Explore correlations between metrics

## Configuration

### Environment Variables
| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | Yes |
| `DATA_PATH` | Path to sales data CSV | No (default provided) |

### Customization
- Modify `styles.css` for UI changes
- Adjust visualization parameters in `visualizations.py`
- Update authentication settings in `auth.py`

## Performance Optimization

### Caching
- Data loading and processing cached for efficiency
- Visualizations cached to reduce computation

### Data Handling
- Optimized pandas operations
- Memory-efficient processing

## Security

### Authentication
- Password hashing with SHA-256
- Session-based authentication
- Role-based access control

### Data Protection
- Local data storage encryption
- Secure API key handling



## Future Enhancements
- Real-time data integration
- Enhanced predictive modeling
- Custom report generation
- Mobile optimization
- Additional authentication providers

