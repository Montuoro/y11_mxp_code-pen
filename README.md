# Year 11 Performance Analysis Tool

A comprehensive tool for analyzing Year 11 student performance by comparing expected vs actual scores using regression analysis.

## Features

- Fetches student data from SQL Server database
- Performs regression analysis with interaction terms across subject categories
- Categorizes subjects into: Maths, Science, Humanities, Language, and Creative
- Generates standalone interactive HTML visualizations with Plotly
- Exports results to Excel spreadsheets

## Requirements

- Python 3.7 or higher
- SQL Server ODBC Driver 17
- Active database connection to Psam database

## Installation

1. Ensure Python is installed and in your system PATH
2. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Install ODBC Driver 17 for SQL Server if not already installed:
   - Download from: https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server

## Usage

### Quick Start

Simply double-click `run_analysis.bat` to launch the application.

Alternatively, run from command line:
```
python app.py
```

### Application Workflow

1. **Select School**: Choose a school from the dropdown (loaded automatically from database)

2. **Enter Year 11 Cohort Year**: Enter the calendar year the students are in Year 11 (e.g., 2025)

3. **Enter Run ID**: Specify the forecast run ID (default: 236)

4. **Set Output Folder**: Specify where to save results (default: current directory)

5. **Fetch Data**: Click "Fetch Data" to retrieve student records from the database

6. **Run Analysis**: Click "Run Analysis" to perform regression analysis and generate predictions

7. **Generate Visualization**: Click "Generate Visualization" to create a standalone HTML file

### Outputs

The tool generates the following files in a timestamped folder:

- `predictions.xlsx` - Complete results with expected, actual, and residual scores
- `visualization.html` - Self-contained interactive visualization (single file, no external dependencies except Plotly CDN)

### Visualization

Open the generated `visualization.html` file in any web browser to view:
- **All Students View**: Individual student performance scatter plot across all courses
- **Course Averages View**: Aggregate performance by course
- **Course Drill-Down View**: Focus on a specific course with regression line
- **Student View**: Individual student's performance across all their courses
- Interactive tooltips with detailed information
- Real-time statistical metrics (R², correlation, mean difference)

## Database Configuration

The tool connects to:
- Server: 103.13.102.146
- Database: Psam
- Authentication: SQL Server authentication

To modify database settings, edit the `DB_CONFIG` dictionary in `app.py`.

## Subject Categories

The tool categorizes subjects into five groups:
1. **Maths**: Mathematics Advanced, Standard, Extension 1/2
2. **Science**: Biology, Chemistry, Physics, Earth Science, etc.
3. **Humanities**: History, Geography, Economics, Business, Legal Studies, etc.
4. **Language**: English, Foreign Languages (continuers/extension)
5. **Creative**: Arts, Music, Drama, Technology, VET courses

## Model Information

The regression model uses:
- Base features: Verbal, Non-verbal, Maths percentile, Reading, Spelling, Writing (Year 10 scores)
- Interaction terms: Each Year 10 score × Subject category
- Output: Predicted Year 11 performance

## Troubleshooting

**Database Connection Failed**
- Verify network connectivity to server
- Confirm ODBC Driver 17 is installed
- Check credentials in DB_CONFIG

**No Schools Loaded**
- Ensure database connection is working
- Verify RunResult table has data
- Check SQL query permissions

**Missing Dependencies**
- Run: `pip install -r requirements.txt`
- Ensure all packages install successfully

**Visualization Not Working**
- Open HTML file in a modern browser (Chrome, Firefox, Edge)
- Ensure JavaScript is enabled
- Check browser console for errors

## Support

For issues or questions, contact your database administrator or development team.

## License

Internal use only - Educational institution data analysis tool
