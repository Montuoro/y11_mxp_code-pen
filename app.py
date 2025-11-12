"""
Year 11 Performance Analysis Tool
Fetches data from SQL Server, runs regression analysis, and generates CodePen visualizations
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pyodbc
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

DB_CONFIG = {
    'SERVER': '103.13.102.146',
    'DATABASE': 'Psam',
    'USERNAME': 'ets',
    'PASSWORD': 'C0ntr0ll3r'
}

# =============================================================================
# SUBJECT CATEGORIZATION
# =============================================================================

SUBJECT_CATEGORIES = {
    'Maths': [
        'Mathematics Advanced', 'Mathematics Standard', 'Mathematics Extension 1',
        'Mathematics Extension 2', 'Mathematics', 'Maths'
    ],
    
    'Science': [
        'Biology', 'Chemistry', 'Physics', 'Earth and Environmental Science',
        'Investigating Science', 'Science Extension', 'Psychology'
    ],
    
    'Humanities': [
        'Ancient History', 'Modern History', 'History Extension',
        'Geography', 'Economics', 'Business Studies', 'Legal Studies',
        'Society and Culture', 'Studies of Religion I', 'Studies of Religion II',
        'Studies of Religion 1', 'Studies of Religion 2',
        'Aboriginal Studies', 'Community and Family Studies'
    ],
    
    'Language': [
        'English Advanced', 'English Standard', 'English Extension',
        'English EAL/D', 'English Studies', 'English Language', 'Literature',
        'French', 'German', 'Spanish', 'Italian', 'Japanese', 'Chinese',
        'Indonesian', 'Arabic', 'Korean', 'Latin', 'Greek', 'Hebrew',
        'Vietnamese', 'Hindi', 'Turkish', 'Portuguese', 'Polish',
        'Croatian', 'Serbian', 'Macedonian', 'Dutch', 'Swedish',
        'Persian', 'Filipino', 'Tamil', 'Punjabi', 'Hungarian', 'Armenian',
        'French Continuers', 'Japanese Continuers', 'Chinese Continuers',
        'German Continuers', 'Spanish Continuers', 'Italian Continuers',
        'Indonesian Continuers', 'Korean Continuers', 'Arabic Continuers',
        'French Extension', 'Japanese Extension', 'Chinese Extension',
        'German Extension', 'Italian Extension', 'Spanish Extension',
        'Indonesian Extension', 'Korean Extension', 'Arabic Extension'
    ],
    
    'Creative': [
        'Visual Arts', 'Music 1', 'Music 2', 'Music Extension',
        'Drama', 'Dance', 'Design and Technology', 'Industrial Technology',
        'Food Technology', 'Textiles and Design', 'Agriculture',
        'Engineering Studies', 'Personal Development, Health and Physical Education',
        'PDHPE', 'Health and Movement Science',
        'Photography, Video and Digital Imaging', 'Ceramics', 'Visual Design',
        'Marine Studies', 'Sport, Lifestyle and Recreation Studies',
        'Exploring Early Childhood', 'Enterprise Computing',
        'Software Engineering', 'Computing Applications',
        'VET', 'Hospitality', 'Construction', 'Entertainment Industry',
        'Information and Digital Technology', 'Business Services',
        'Human Services', 'Primary Industries', 'Automotive',
        'Electrotechnology', 'Financial Services', 'Tourism, Travel and Events',
        'Retail Services', 'Skills for Work and Vocational Pathways'
    ]
}

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def categorize_subject(course_name):
    """Categorize a course into one of the five subject categories."""
    course_lower = course_name.lower()
    
    for category, subjects in SUBJECT_CATEGORIES.items():
        for subject in subjects:
            if subject.lower() in course_lower:
                return category
    
    # Default fallback
    if any(word in course_lower for word in ['math', 'calculus', 'algebra']):
        return 'Maths'
    elif any(word in course_lower for word in ['history', 'geography', 'economics', 'business', 'legal', 'studies']):
        return 'Humanities'
    elif any(word in course_lower for word in ['english', 'language', 'literature']):
        return 'Language'
    elif any(word in course_lower for word in ['biology', 'chemistry', 'physics', 'science']):
        return 'Science'
    else:
        return 'Creative'

def create_interaction_features(df):
    """Create interaction terms between Year 10 scores and subject categories."""
    
    # Create binary columns for each category
    for category in ['Maths', 'Science', 'Humanities', 'Language', 'Creative']:
        df[f'is_{category}'] = (df['category'] == category).astype(int)
    
    # Create interaction terms
    year10_features = ['v', 'n', 'm', 'r', 's', 'w']
    
    for feature in year10_features:
        for category in ['Maths', 'Science', 'Humanities', 'Language', 'Creative']:
            df[f'{feature}_x_{category}'] = df[feature] * df[f'is_{category}']
    
    return df

def build_model(df):
    """Build regression model with interaction terms."""
    
    # Base features
    base_features = ['v', 'n', 'm', 'r', 's', 'w']
    
    # Interaction features
    interaction_features = [col for col in df.columns if '_x_' in col]
    
    # All features
    all_features = base_features + interaction_features
    
    # Prepare data
    X = df[all_features]
    y = df['actual']
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Calculate metrics
    r2 = r2_score(y, predictions)
    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(np.mean((y - predictions) ** 2))
    
    return model, all_features, predictions, r2, mae, rmse

# =============================================================================
# MAIN APPLICATION CLASS
# =============================================================================

class Year11AnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Year 11 Performance Analysis Tool")
        self.root.geometry("900x700")

        # Data storage
        self.df = None
        self.school_name = ""

        # Set up GUI
        self.setup_gui()

        # Load schools on startup
        self.load_schools()
    
    def setup_gui(self):
        """Set up the GUI components."""
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="📊 Year 11 Performance Analysis Tool", 
                                font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # School selection
        ttk.Label(main_frame, text="Select School:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.school_combo = ttk.Combobox(main_frame, width=40)
        self.school_combo.grid(row=1, column=1, padx=(10, 0), pady=5)
        self.school_combo.bind('<<ComboboxSelected>>', self.load_year_levels)
        
        # Year selection
        ttk.Label(main_frame, text="Year 11 Cohort Year:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.year_entry = ttk.Entry(main_frame, width=10)
        self.year_entry.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        self.year_entry.insert(0, "2025")
        
        # Add explanation label
        year_note = ttk.Label(main_frame, text="(e.g., enter 2025 for current Year 11s)", 
                             font=('Arial', 9), foreground='gray')
        year_note.grid(row=2, column=1, sticky=tk.W, padx=(120, 0), pady=5)
        
        # Study Year selection
        ttk.Label(main_frame, text="Study Year:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.year_level_combo = ttk.Combobox(main_frame, width=20, state='readonly')
        self.year_level_combo.grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        self.year_level_combo.bind('<<ComboboxSelected>>', self.load_run_dates)

        # Run Date selection
        ttk.Label(main_frame, text="Run Date:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.run_date_combo = ttk.Combobox(main_frame, width=30, state='readonly')
        self.run_date_combo.grid(row=4, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        # Chart title
        ttk.Label(main_frame, text="Chart Title:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.chart_title_entry = ttk.Entry(main_frame, width=50)
        self.chart_title_entry.grid(row=5, column=1, padx=(10, 0), pady=5)
        self.chart_title_entry.insert(0, "Year 11 Performance Analysis")

        # Output folder
        ttk.Label(main_frame, text="Output Folder:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.output_entry = ttk.Entry(main_frame, width=50)
        self.output_entry.grid(row=6, column=1, padx=(10, 0), pady=5)
        self.output_entry.insert(0, r"C:\Data Projects\python\y11_mxp_code-pen")
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=7, column=0, columnspan=2, pady=20)

        ttk.Button(button_frame, text="🔍 Fetch Data", command=self.fetch_data).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="📈 Run Analysis", command=self.run_analysis).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="💾 Generate CodePen Files", command=self.generate_codepen).grid(row=0, column=2, padx=5)

        # Progress/Status area
        ttk.Label(main_frame, text="Status:").grid(row=8, column=0, sticky=tk.W, pady=(10, 5))
        self.status_text = scrolledtext.ScrolledText(main_frame, height=15, width=100)
        self.status_text.grid(row=9, column=0, columnspan=2, pady=5)

        # Statistics display
        self.stats_frame = ttk.LabelFrame(main_frame, text="Model Statistics", padding="10")
        self.stats_frame.grid(row=10, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
    
    def log_message(self, message):
        """Add a message to the status text."""
        self.status_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
        self.status_text.see(tk.END)
        self.root.update()
    
    def load_schools(self):
        """Load available schools from the database."""
        try:
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={DB_CONFIG['SERVER']};"
                f"DATABASE={DB_CONFIG['DATABASE']};"
                f"UID={DB_CONFIG['USERNAME']};"
                f"PWD={DB_CONFIG['PASSWORD']}"
            )
            
            conn = pyodbc.connect(conn_str)
            cursor = conn.cursor()
            
            # Get unique schools
            # First, check what columns exist in the RunResult table
            column_check_query = """
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = 'dbo'
                    AND TABLE_NAME = 'RunResult'
                    AND TABLE_CATALOG = 'Forecast'
            """

            cursor.execute(column_check_query)
            columns = [row[0] for row in cursor.fetchall()]
            self.log_message(f"Available columns in RunResult: {', '.join(columns)}")

            # Determine which column to use for school name
            school_name_column = None
            possible_names = ['SchoolName', 'School_Name', 'Name', 'StudentSchool', 'School']

            for possible_name in possible_names:
                if possible_name in columns:
                    school_name_column = possible_name
                    break

            # Build the query based on available columns
            if school_name_column:
                query = f"""
                    SELECT DISTINCT SchoolId, {school_name_column} as SchoolName
                    FROM Forecast.dbo.RunResult
                    WHERE SchoolId IS NOT NULL
                    ORDER BY {school_name_column}
                """
            else:
                # Fallback: just use SchoolId if no name column exists
                self.log_message("Warning: No school name column found, using SchoolId only")
                query = """
                    SELECT DISTINCT SchoolId, CAST(SchoolId AS VARCHAR) as SchoolName
                    FROM Forecast.dbo.RunResult
                    WHERE SchoolId IS NOT NULL
                    ORDER BY SchoolId
                """

            cursor.execute(query)
            schools = cursor.fetchall()

            school_list = [f"{row[1]} (ID: {row[0]})" for row in schools]
            self.school_combo['values'] = school_list

            conn.close()

            self.log_message(f"Loaded {len(school_list)} schools from database")
            
        except Exception as e:
            self.log_message(f"Error loading schools: {str(e)}")
            messagebox.showerror("Database Error", f"Could not load schools: {str(e)}")

    def load_year_levels(self, event=None):
        """Load available study years for selected school."""
        try:
            school_selection = self.school_combo.get()
            if not school_selection:
                return

            school_id = int(school_selection.split("ID: ")[1].replace(")", ""))

            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={DB_CONFIG['SERVER']};"
                f"DATABASE={DB_CONFIG['DATABASE']};"
                f"UID={DB_CONFIG['USERNAME']};"
                f"PWD={DB_CONFIG['PASSWORD']}"
            )

            conn = pyodbc.connect(conn_str)
            cursor = conn.cursor()

            # Query to get distinct study years for this school
            query = f"""
                SELECT DISTINCT StudyYear
                FROM Forecast.dbo.RunResult
                WHERE SchoolId = {school_id}
                AND StudyYear IS NOT NULL
                ORDER BY StudyYear
            """

            cursor.execute(query)
            year_levels = [str(row[0]) for row in cursor.fetchall()]

            conn.close()

            self.year_level_combo['values'] = year_levels
            if year_levels:
                self.year_level_combo.current(0)
                self.load_run_dates()

            self.log_message(f"Loaded {len(year_levels)} study years")

        except Exception as e:
            self.log_message(f"Error loading study years: {str(e)}")

    def load_run_dates(self, event=None):
        """Load available run dates for selected school and study year."""
        try:
            school_selection = self.school_combo.get()
            year_level = self.year_level_combo.get()

            if not school_selection or not year_level:
                return

            school_id = int(school_selection.split("ID: ")[1].replace(")", ""))

            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={DB_CONFIG['SERVER']};"
                f"DATABASE={DB_CONFIG['DATABASE']};"
                f"UID={DB_CONFIG['USERNAME']};"
                f"PWD={DB_CONFIG['PASSWORD']}"
            )

            conn = pyodbc.connect(conn_str)
            cursor = conn.cursor()

            # First, check if there's a Run table we can join
            # Try to get run information from a Run table if it exists
            query = f"""
                SELECT DISTINCT
                    rr.RunId,
                    COALESCE(r.RunDate, r.Created, GETDATE()) as RunDate
                FROM Forecast.dbo.RunResult rr
                LEFT JOIN Forecast.dbo.Run r ON r.Id = rr.RunId
                WHERE rr.SchoolId = {school_id}
                AND rr.StudyYear = {year_level}
                ORDER BY RunDate DESC, rr.RunId DESC
            """

            try:
                cursor.execute(query)
                runs = cursor.fetchall()

                # Format as "YYYY-MM-DD (RunId: XXX)"
                run_list = [f"{row[1].strftime('%Y-%m-%d')} (RunId: {row[0]})" for row in runs]
            except Exception as e:
                # If Run table doesn't exist or join fails, just list RunIds
                self.log_message(f"Note: Using RunId only (Run table not found or accessible)")
                query = f"""
                    SELECT DISTINCT RunId
                    FROM Forecast.dbo.RunResult
                    WHERE SchoolId = {school_id}
                    AND StudyYear = {year_level}
                    ORDER BY RunId DESC
                """
                cursor.execute(query)
                runs = cursor.fetchall()
                run_list = [f"RunId: {row[0]}" for row in runs]

            conn.close()

            self.run_date_combo['values'] = run_list

            if run_list:
                self.run_date_combo.current(0)

            self.log_message(f"Loaded {len(run_list)} runs for Study Year {year_level}")

        except Exception as e:
            self.log_message(f"Error loading run dates: {str(e)}")

    def fetch_data(self):
        """Fetch data from SQL Server based on selected parameters."""
        try:
            # Get school ID from selection
            school_selection = self.school_combo.get()
            if not school_selection:
                messagebox.showwarning("Input Error", "Please select a school")
                return

            # Get study year
            year_level = self.year_level_combo.get()
            if not year_level:
                messagebox.showwarning("Input Error", "Please select a study year")
                return

            # Get run date selection
            run_date_selection = self.run_date_combo.get()
            if not run_date_selection:
                messagebox.showwarning("Input Error", "Please select a run date")
                return

            school_id = int(school_selection.split("ID: ")[1].replace(")", ""))
            self.school_name = school_selection.split(" (ID:")[0]
            year_11_cohort = int(self.year_entry.get())
            # Database stores Year 11 students under their Year 12 calendar year
            db_calendar_year = year_11_cohort + 1

            # Extract RunId from run date selection (handles both "YYYY-MM-DD (RunId: XXX)" and "RunId: XXX" formats)
            if "RunId: " in run_date_selection:
                run_id = int(run_date_selection.split("RunId: ")[1].replace(")", ""))
            else:
                # Fallback in case format is unexpected
                run_id = int(run_date_selection)
            
            self.log_message(f"Fetching data for {self.school_name} (ID: {school_id})")
            self.log_message(f"Study Year: {year_level}, Run: {run_date_selection}")
            self.log_message(f"Year 11 Cohort: {year_11_cohort} (DB Calendar Year: {db_calendar_year})")
            
            # Connect to database
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={DB_CONFIG['SERVER']};"
                f"DATABASE={DB_CONFIG['DATABASE']};"
                f"UID={DB_CONFIG['USERNAME']};"
                f"PWD={DB_CONFIG['PASSWORD']}"
            )
            
            conn = pyodbc.connect(conn_str)
            
            # Build and execute query
            query = f"""
                SELECT 
                    rr.StudentName AS student_name,
                    rr.StudentId AS student_number,
                    rr.SubjectName AS course,
                    ar.verbal AS v,
                    ar.nonverbal AS n,
                    ar.mathper AS m,
                    ar.reading AS r,
                    ar.spelling AS s,
                    ar.writing AS w,
                    rr.Prediction AS actual
                FROM Forecast.dbo.RunResult rr
                INNER JOIN Psam.dbo.AasResult ar
                    ON ar.SchoolId = {school_id}
                    AND ar.CalendarYear = {db_calendar_year}
                    AND (
                        UPPER(rr.StudentName) COLLATE Latin1_General_CI_AS = UPPER(ar.Name) COLLATE Latin1_General_CI_AS
                        OR
                        REPLACE(UPPER(rr.StudentName), ' ', '') COLLATE Latin1_General_CI_AS = REPLACE(UPPER(ar.Name), ' ', '') COLLATE Latin1_General_CI_AS
                        OR
                        (
                            UPPER(LEFT(rr.StudentName, CHARINDEX(' ', rr.StudentName) - 1)) COLLATE Latin1_General_CI_AS = 
                            UPPER(LEFT(ar.Name, CHARINDEX(' ', ar.Name) - 1)) COLLATE Latin1_General_CI_AS
                            AND
                            UPPER(SUBSTRING(rr.StudentName, CHARINDEX(' ', rr.StudentName) + 1, 4)) COLLATE Latin1_General_CI_AS = 
                            UPPER(SUBSTRING(ar.Name, CHARINDEX(' ', ar.Name) + 1, 4)) COLLATE Latin1_General_CI_AS
                        )
                    )
                WHERE rr.RunId = {run_id}
                    AND rr.SchoolId = {school_id}
                    AND rr.Prediction IS NOT NULL
                ORDER BY student_name, course
            """
            
            self.df = pd.read_sql_query(query, conn)
            conn.close()
            
            self.log_message(f"Fetched {len(self.df)} records for {self.df['student_name'].nunique()} students")
            self.log_message(f"Courses: {self.df['course'].nunique()}")
            
        except Exception as e:
            self.log_message(f"Error fetching data: {str(e)}")
            messagebox.showerror("Database Error", f"Could not fetch data: {str(e)}")
    
    def run_analysis(self):
        """Run the regression analysis on fetched data."""
        if self.df is None:
            messagebox.showwarning("No Data", "Please fetch data first")
            return

        try:
            self.log_message("Starting regression analysis...")

            # Categorize subjects
            self.df['category'] = self.df['course'].apply(categorize_subject)

            # Log category distribution
            cat_dist = self.df['category'].value_counts()
            self.log_message("\nSubject Category Distribution:")
            for cat, count in cat_dist.items():
                self.log_message(f"  {cat}: {count}")

            # Create interaction features
            self.df = create_interaction_features(self.df)

            # Build model
            model, features, predictions, r2, mae, rmse = build_model(self.df)

            # Add predictions to dataframe
            self.df['expected'] = predictions
            self.df['residual'] = self.df['actual'] - self.df['expected']

            # Display statistics
            self.display_statistics(r2, mae, rmse)

            # Create timestamped output folder
            base_folder = self.output_entry.get()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            school_folder_name = f"{self.school_name.replace(' ', '_')}_{timestamp}"
            output_folder = os.path.join(base_folder, school_folder_name)
            os.makedirs(output_folder, exist_ok=True)

            # Store output folder for use in generate_codepen
            self.current_output_folder = output_folder

            # Save full results
            results_file = os.path.join(output_folder, "predictions.xlsx")
            output_df = self.df[['student_name', 'student_number', 'course', 'expected', 'actual', 'residual']].copy()
            output_df['expected'] = output_df['expected'].round(2)
            output_df['residual'] = output_df['residual'].round(2)
            output_df.to_excel(results_file, index=False)

            self.log_message(f"\n✅ Analysis complete! Results saved to {output_folder}")

        except Exception as e:
            self.log_message(f"Error running analysis: {str(e)}")
            messagebox.showerror("Analysis Error", f"Could not complete analysis: {str(e)}")
    
    def display_statistics(self, r2, mae, rmse):
        """Display model statistics in the GUI."""
        # Clear previous stats
        for widget in self.stats_frame.winfo_children():
            widget.destroy()
        
        stats = [
            ("R² Score", f"{r2:.4f}"),
            ("Mean Absolute Error", f"{mae:.2f}"),
            ("Root Mean Squared Error", f"{rmse:.2f}"),
            ("Total Students", f"{self.df['student_name'].nunique()}"),
            ("Total Records", f"{len(self.df)}")
        ]
        
        for i, (label, value) in enumerate(stats):
            ttk.Label(self.stats_frame, text=f"{label}:").grid(row=i//3, column=(i%3)*2, sticky=tk.W, padx=5, pady=2)
            ttk.Label(self.stats_frame, text=value, font=('Arial', 10, 'bold')).grid(row=i//3, column=(i%3)*2+1, sticky=tk.W, padx=5, pady=2)
        
        self.log_message(f"\nModel Performance: R² = {r2:.4f}, MAE = {mae:.2f}, RMSE = {rmse:.2f}")
    
    def generate_codepen(self):
        """Generate HTML, CSS, and JS files for CodePen visualization."""
        if self.df is None or 'expected' not in self.df.columns:
            messagebox.showwarning("No Analysis", "Please run analysis first")
            return

        try:
            # Use the stored output folder from run_analysis
            if not hasattr(self, 'current_output_folder'):
                messagebox.showwarning("No Analysis", "Please run analysis first to create output folder")
                return

            output_folder = self.current_output_folder

            self.log_message("Generating CodePen files...")

            # Prepare data for JavaScript
            data_for_js = self.df[['student_name', 'course', 'expected', 'actual']].to_dict('records')

            # Calculate course means
            course_means = self.df.groupby('course').agg({
                'expected': 'mean',
                'actual': 'mean'
            }).round(2).reset_index()
            course_means_data = course_means.to_dict('records')

            # Get unique courses for dropdown
            unique_courses = sorted(self.df['course'].unique().tolist())

            # Get chart title
            chart_title = self.chart_title_entry.get() or "Performance Analysis"

            # Generate JavaScript file
            js_content = self.generate_js_content(data_for_js, course_means_data, unique_courses, chart_title)

            # Generate CSS file
            css_content = self.generate_css_content()

            # Generate HTML file
            html_content = self.generate_html_content(chart_title)

            # Save files
            js_file = os.path.join(output_folder, "viz.js")
            css_file = os.path.join(output_folder, "viz.css")
            html_file = os.path.join(output_folder, "viz.html")

            with open(js_file, 'w', encoding='utf-8') as f:
                f.write(js_content)

            with open(css_file, 'w', encoding='utf-8') as f:
                f.write(css_content)

            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)

            self.log_message(f"✅ CodePen files generated successfully!")
            self.log_message(f"  - JavaScript: {js_file}")
            self.log_message(f"  - CSS: {css_file}")
            self.log_message(f"  - HTML: {html_file}")

            messagebox.showinfo("Success", "CodePen files generated successfully!")

        except Exception as e:
            self.log_message(f"Error generating CodePen files: {str(e)}")
            messagebox.showerror("Generation Error", f"Could not generate files: {str(e)}")
    
    def generate_js_content(self, data, course_means, courses, chart_title):
        """Generate JavaScript content for visualization."""
        js_template = """// Performance Analysis Visualization for {school_name}
// Generated on {date}

const rawData = {data_json};
const courseMeans = {means_json};
const allCourses = {courses_json};
const allStudents = {students_json};
const chartTitle = {chart_title_json};

let currentView = 'individual';
let selectedCourse = null;
let selectedStudent = null;

// Calculate statistics
function calculateStats(data) {{
    const pairs = data.map(d => [d.expected, d.actual]);
    const n = pairs.length;

    // Calculate means
    const meanX = pairs.reduce((sum, p) => sum + p[0], 0) / n;
    const meanY = pairs.reduce((sum, p) => sum + p[1], 0) / n;

    // Calculate correlation
    let sumXY = 0, sumX2 = 0, sumY2 = 0;
    pairs.forEach(p => {{
        const dx = p[0] - meanX;
        const dy = p[1] - meanY;
        sumXY += dx * dy;
        sumX2 += dx * dx;
        sumY2 += dy * dy;
    }});

    const correlation = sumXY / Math.sqrt(sumX2 * sumY2);
    const r2 = correlation * correlation;

    const meanDiff = (meanY - meanX).toFixed(2);

    return {{
        totalPoints: n,
        r2: r2.toFixed(4),
        correlation: correlation.toFixed(4),
        meanDiff: meanDiff
    }};
}}

// Create individual student plot
function createIndividualPlot() {{
    const trace = {{
        x: rawData.map(d => d.expected),
        y: rawData.map(d => d.actual),
        mode: 'markers',
        type: 'scatter',
        text: rawData.map(d => `${{d.student_name}}<br>${{d.course}}<br>Expected: ${{d.expected}}<br>Forecasted: ${{d.actual}}`),
        hoverinfo: 'text',
        marker: {{
            size: 8,
            color: rawData.map(d => d.actual - d.expected),
            colorscale: 'RdYlGn',
            colorbar: {{
                title: 'Difference<br>(Forecasted - Expected)',
                thickness: 20
            }},
            showscale: true
        }}
    }};

    // Add perfect correlation line
    const minVal = Math.min(...rawData.map(d => Math.min(d.expected, d.actual)));
    const maxVal = Math.max(...rawData.map(d => Math.max(d.expected, d.actual)));

    const line = {{
        x: [minVal, maxVal],
        y: [minVal, maxVal],
        mode: 'lines',
        type: 'scatter',
        line: {{
            color: 'rgba(255, 0, 0, 0.5)',
            width: 2,
            dash: 'dash'
        }},
        hoverinfo: 'skip',
        showlegend: false
    }};

    const layout = {{
        title: chartTitle + ': All Students',
        xaxis: {{
            title: 'Expected Score',
            range: [minVal - 5, maxVal + 5]
        }},
        yaxis: {{
            title: 'Forecasted',
            range: [minVal - 5, maxVal + 5]
        }},
        hovermode: 'closest',
        height: 600,
        showlegend: false
    }};

    Plotly.newPlot('plot', [trace, line], layout);
}}

// Create course means plot
function createMeansPlot() {{
    const trace = {{
        x: courseMeans.map(d => d.expected),
        y: courseMeans.map(d => d.actual),
        mode: 'markers+text',
        type: 'scatter',
        text: courseMeans.map(d => d.course),
        textposition: 'top center',
        textfont: {{
            size: 10
        }},
        hovertext: courseMeans.map(d => `${{d.course}}<br>Mean Expected: ${{d.expected.toFixed(2)}}<br>Mean Forecasted: ${{d.actual.toFixed(2)}}`),
        hoverinfo: 'text',
        marker: {{
            size: 12,
            color: courseMeans.map(d => d.actual - d.expected),
            colorscale: 'RdYlGn',
            colorbar: {{
                title: 'Mean Difference<br>(Forecasted - Expected)',
                thickness: 20
            }},
            showscale: true
        }}
    }};

    const minVal = Math.min(...courseMeans.map(d => Math.min(d.expected, d.actual)));
    const maxVal = Math.max(...courseMeans.map(d => Math.max(d.expected, d.actual)));

    const line = {{
        x: [minVal, maxVal],
        y: [minVal, maxVal],
        mode: 'lines',
        type: 'scatter',
        line: {{
            color: 'rgba(255, 0, 0, 0.5)',
            width: 2,
            dash: 'dash'
        }},
        hoverinfo: 'skip',
        showlegend: false
    }};

    const layout = {{
        title: chartTitle + ': Course Averages',
        xaxis: {{
            title: 'Mean Expected Score',
            range: [minVal - 5, maxVal + 5]
        }},
        yaxis: {{
            title: 'Mean Forecasted',
            range: [minVal - 5, maxVal + 5]
        }},
        hovermode: 'closest',
        height: 600,
        showlegend: false
    }};

    Plotly.newPlot('plot', [trace, line], layout);
}}

// Calculate linear regression
function calculateRegression(data) {{
    const n = data.length;
    const xData = data.map(d => d.expected);
    const yData = data.map(d => d.actual);

    const sumX = xData.reduce((a, b) => a + b, 0);
    const sumY = yData.reduce((a, b) => a + b, 0);
    const sumXY = xData.reduce((sum, x, i) => sum + x * yData[i], 0);
    const sumX2 = xData.reduce((sum, x) => sum + x * x, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    return {{ slope, intercept }};
}}

// Create course drill-down plot
function createCoursePlot(course) {{
    // Filter data for selected course
    const courseData = rawData.filter(d => d.course === course);
    const backgroundData = rawData.filter(d => d.course !== course);

    // Background trace (greyed out) - 40% larger
    const bgTrace = {{
        x: backgroundData.map(d => d.expected),
        y: backgroundData.map(d => d.actual),
        mode: 'markers',
        type: 'scatter',
        text: backgroundData.map(d => `${{d.student_name}}<br>${{d.course}}<br>Expected: ${{d.expected}}<br>Forecasted: ${{d.actual}}`),
        hoverinfo: 'text',
        marker: {{
            size: 8.4,
            color: 'rgba(200, 200, 200, 0.3)',
            line: {{
                color: 'rgba(150, 150, 150, 0.3)',
                width: 1
            }}
        }},
        showlegend: false
    }};

    // Course-specific trace (highlighted)
    const courseTrace = {{
        x: courseData.map(d => d.expected),
        y: courseData.map(d => d.actual),
        mode: 'markers',
        type: 'scatter',
        text: courseData.map(d => `${{d.student_name}}<br>${{d.course}}<br>Expected: ${{d.expected}}<br>Forecasted: ${{d.actual}}`),
        hoverinfo: 'text',
        marker: {{
            size: 10,
            color: courseData.map(d => d.actual - d.expected),
            colorscale: 'RdYlGn',
            colorbar: {{
                title: 'Difference<br>(Forecasted - Expected)',
                thickness: 20
            }},
            showscale: true,
            line: {{
                color: 'rgba(0, 0, 0, 0.5)',
                width: 1
            }}
        }},
        showlegend: false
    }};

    // Calculate regression line for course data
    const regression = calculateRegression(courseData);
    const minX = Math.min(...courseData.map(d => d.expected));
    const maxX = Math.max(...courseData.map(d => d.expected));

    const regressionTrace = {{
        x: [minX, maxX],
        y: [regression.slope * minX + regression.intercept,
            regression.slope * maxX + regression.intercept],
        mode: 'lines',
        type: 'scatter',
        line: {{
            color: 'rgba(0, 100, 255, 0.8)',
            width: 3
        }},
        showlegend: false,
        hoverinfo: 'skip'
    }};

    // Add perfect correlation line
    const minVal = Math.min(...rawData.map(d => Math.min(d.expected, d.actual)));
    const maxVal = Math.max(...rawData.map(d => Math.max(d.expected, d.actual)));

    const line = {{
        x: [minVal, maxVal],
        y: [minVal, maxVal],
        mode: 'lines',
        type: 'scatter',
        line: {{
            color: 'rgba(255, 0, 0, 0.5)',
            width: 2,
            dash: 'dash'
        }},
        hoverinfo: 'skip',
        showlegend: false
    }};

    const layout = {{
        title: chartTitle + `: ${{course}}`,
        xaxis: {{
            title: 'Expected Score',
            range: [minVal - 5, maxVal + 5]
        }},
        yaxis: {{
            title: 'Forecasted',
            range: [minVal - 5, maxVal + 5]
        }},
        hovermode: 'closest',
        height: 600,
        showlegend: false
    }};

    Plotly.newPlot('plot', [bgTrace, courseTrace, regressionTrace, line], layout);
}}

// Create student drill-down plot
function createStudentPlot(studentName) {{
    // Filter data for selected student
    const studentData = rawData.filter(d => d.student_name === studentName);
    const backgroundData = rawData.filter(d => d.student_name !== studentName);

    // Background trace (greyed out) - 40% larger
    const bgTrace = {{
        x: backgroundData.map(d => d.expected),
        y: backgroundData.map(d => d.actual),
        mode: 'markers',
        type: 'scatter',
        text: backgroundData.map(d => `${{d.student_name}}<br>${{d.course}}<br>Expected: ${{d.expected}}<br>Forecasted: ${{d.actual}}`),
        hoverinfo: 'text',
        marker: {{
            size: 8.4,
            color: 'rgba(200, 200, 200, 0.3)',
            line: {{
                color: 'rgba(150, 150, 150, 0.3)',
                width: 1
            }}
        }},
        showlegend: false
    }};

    // Student-specific trace (highlighted)
    const studentTrace = {{
        x: studentData.map(d => d.expected),
        y: studentData.map(d => d.actual),
        mode: 'markers',
        type: 'scatter',
        text: studentData.map(d => `${{d.student_name}}<br>${{d.course}}<br>Expected: ${{d.expected}}<br>Forecasted: ${{d.actual}}`),
        hoverinfo: 'text',
        marker: {{
            size: 12,
            color: studentData.map(d => d.actual - d.expected),
            colorscale: 'RdYlGn',
            colorbar: {{
                title: 'Difference<br>(Forecasted - Expected)',
                thickness: 20
            }},
            showscale: true,
            line: {{
                color: 'rgba(0, 0, 0, 0.8)',
                width: 2
            }}
        }},
        showlegend: false
    }};

    // Add perfect correlation line
    const minVal = Math.min(...rawData.map(d => Math.min(d.expected, d.actual)));
    const maxVal = Math.max(...rawData.map(d => Math.max(d.expected, d.actual)));

    const line = {{
        x: [minVal, maxVal],
        y: [minVal, maxVal],
        mode: 'lines',
        type: 'scatter',
        line: {{
            color: 'rgba(255, 0, 0, 0.5)',
            width: 2,
            dash: 'dash'
        }},
        hoverinfo: 'skip',
        showlegend: false
    }};

    const layout = {{
        title: chartTitle + `: ${{studentName}}`,
        xaxis: {{
            title: 'Expected Score',
            range: [minVal - 5, maxVal + 5]
        }},
        yaxis: {{
            title: 'Forecasted',
            range: [minVal - 5, maxVal + 5]
        }},
        hovermode: 'closest',
        height: 600,
        showlegend: false
    }};

    Plotly.newPlot('plot', [bgTrace, studentTrace, line], layout);
}}

// Update statistics display
function updateStats() {{
    let statsData;
    if (currentView === 'individual') {{
        statsData = rawData;
    }} else if (currentView === 'means') {{
        statsData = courseMeans;
    }} else if (currentView === 'course' && selectedCourse) {{
        statsData = rawData.filter(d => d.course === selectedCourse);
    }} else if (currentView === 'student' && selectedStudent) {{
        statsData = rawData.filter(d => d.student_name === selectedStudent);
    }}

    const stats = calculateStats(statsData);
    document.getElementById('totalPoints').textContent = stats.totalPoints;
    document.getElementById('r2Value').textContent = stats.r2;
    document.getElementById('correlation').textContent = stats.correlation;
    document.getElementById('meanDiff').textContent = stats.meanDiff;
}}

// Initialize
document.addEventListener('DOMContentLoaded', function() {{
    // Populate course dropdown
    const dropdown = document.getElementById('course-dropdown');
    allCourses.forEach(course => {{
        const option = document.createElement('option');
        option.value = course;
        option.textContent = course;
        dropdown.appendChild(option);
    }});

    // Set initial selected course
    if (allCourses.length > 0) {{
        selectedCourse = allCourses[0];
    }}

    // Populate student dropdown
    const studentDropdown = document.getElementById('student-dropdown');
    allStudents.forEach(student => {{
        const option = document.createElement('option');
        option.value = student;
        option.textContent = student;
        studentDropdown.appendChild(option);
    }});

    // Set initial selected student
    if (allStudents.length > 0) {{
        selectedStudent = allStudents[0];
    }}

    // Create initial plot
    createIndividualPlot();
    updateStats();

    // Add button event listeners
    document.getElementById('btn-individual').addEventListener('click', function() {{
        document.querySelectorAll('.view-btn').forEach(btn => btn.classList.remove('active'));
        this.classList.add('active');
        document.getElementById('course-selector').style.display = 'none';
        document.getElementById('student-selector').style.display = 'none';
        currentView = 'individual';
        createIndividualPlot();
        updateStats();
    }});

    document.getElementById('btn-means').addEventListener('click', function() {{
        document.querySelectorAll('.view-btn').forEach(btn => btn.classList.remove('active'));
        this.classList.add('active');
        document.getElementById('course-selector').style.display = 'none';
        document.getElementById('student-selector').style.display = 'none';
        currentView = 'means';
        createMeansPlot();
        updateStats();
    }});

    document.getElementById('btn-course').addEventListener('click', function() {{
        document.querySelectorAll('.view-btn').forEach(btn => btn.classList.remove('active'));
        this.classList.add('active');
        document.getElementById('course-selector').style.display = 'block';
        document.getElementById('student-selector').style.display = 'none';
        currentView = 'course';
        createCoursePlot(selectedCourse);
        updateStats();
    }});

    document.getElementById('btn-student').addEventListener('click', function() {{
        document.querySelectorAll('.view-btn').forEach(btn => btn.classList.remove('active'));
        this.classList.add('active');
        document.getElementById('course-selector').style.display = 'none';
        document.getElementById('student-selector').style.display = 'block';
        currentView = 'student';
        createStudentPlot(selectedStudent);
        updateStats();
    }});

    // Add dropdown change listener
    dropdown.addEventListener('change', function() {{
        selectedCourse = this.value;
        if (currentView === 'course') {{
            createCoursePlot(selectedCourse);
            updateStats();
        }}
    }});

    // Add student dropdown change listener
    studentDropdown.addEventListener('change', function() {{
        selectedStudent = this.value;
        if (currentView === 'student') {{
            createStudentPlot(selectedStudent);
            updateStats();
        }}
    }});
}});"""

        # Get unique students sorted alphabetically
        unique_students = sorted(list(set([d['student_name'] for d in data])))

        return js_template.format(
            school_name=self.school_name,
            date=datetime.now().strftime('%Y-%m-%d'),
            data_json=json.dumps(data),
            means_json=json.dumps(course_means),
            courses_json=json.dumps(courses),
            students_json=json.dumps(unique_students),
            chart_title_json=json.dumps(chart_title)
        )
    
    def generate_css_content(self):
        """Generate CSS content for visualization."""
        return """body {
  font-family: Arial, sans-serif;
  margin: 0;
  padding: 20px;
  background: linear-gradient(135deg, #667eea, #764ba2);
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  background: white;
  border-radius: 15px;
  padding: 30px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}

h1 {
  text-align: center;
  color: #333;
  margin-bottom: 10px;
}

.subtitle {
  text-align: center;
  color: #666;
  margin-bottom: 20px;
}

.view-controls {
  display: flex;
  justify-content: center;
  gap: 10px;
  margin-bottom: 20px;
}

.view-btn {
  padding: 10px 20px;
  border: none;
  border-radius: 5px;
  background: #ddd;
  cursor: pointer;
  font-size: 14px;
  font-weight: bold;
  transition: all 0.3s;
}

.view-btn:hover {
  background: #bbb;
}

.view-btn.active {
  background: #667eea;
  color: white;
}

#course-selector, #student-selector {
  text-align: center;
  margin-bottom: 20px;
  padding: 15px;
  background: #f0f0f0;
  border-radius: 8px;
}

#course-selector label, #student-selector label {
  font-weight: bold;
  margin-right: 10px;
  color: #333;
}

#course-dropdown, #student-dropdown {
  padding: 8px 15px;
  border: 2px solid #667eea;
  border-radius: 5px;
  font-size: 14px;
  background: white;
  cursor: pointer;
  min-width: 300px;
}

#course-dropdown:focus, #student-dropdown:focus {
  outline: none;
  border-color: #764ba2;
  box-shadow: 0 0 5px rgba(102, 126, 234, 0.5);
}

#plot {
  width: 100%;
  height: 600px;
}

.stats {
  display: flex;
  justify-content: space-around;
  margin-top: 20px;
  padding: 15px;
  background: #f5f5f5;
  border-radius: 10px;
}

.stat-item {
  text-align: center;
}

.stat-label {
  color: #666;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 1px;
}

.stat-value {
  color: #333;
  font-size: 24px;
  font-weight: bold;
}"""
    
    def generate_html_content(self, chart_title):
        """Generate HTML content for visualization."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.school_name} - {chart_title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div class="container">
        <h1>📊 {chart_title}</h1>
        <div class="subtitle">Expected vs Forecasted Scores Across All Subjects</div>

        <div class="view-controls">
            <button class="view-btn active" id="btn-individual">All Students</button>
            <button class="view-btn" id="btn-means">Course Averages</button>
            <button class="view-btn" id="btn-course">Course Drill-Down</button>
            <button class="view-btn" id="btn-student">Student View</button>
        </div>

        <div id="course-selector" style="display: none;">
            <label for="course-dropdown">Select Course: </label>
            <select id="course-dropdown"></select>
        </div>

        <div id="student-selector" style="display: none;">
            <label for="student-dropdown">Select Student: </label>
            <select id="student-dropdown"></select>
        </div>

        <div id="plot"></div>

        <div class="stats">
            <div class="stat-item">
                <div class="stat-label">Data Points</div>
                <div class="stat-value" id="totalPoints">-</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">R² Value</div>
                <div class="stat-value" id="r2Value">-</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Correlation</div>
                <div class="stat-value" id="correlation">-</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Mean Difference</div>
                <div class="stat-value" id="meanDiff">-</div>
            </div>
        </div>
    </div>
</body>
</html>"""

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    root = tk.Tk()
    app = Year11AnalysisApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()