from graph.workflow import run_workflow
import hashlib
import streamlit as st
import pandas as pd

from tools.document_processor import extract_pdf_text
from agents.document_analyst import generate_document_answer

from tools.data_profiler import (
    get_dataset_overview,
    get_duplicate_records,
    get_missing_values
)

from tools.data_cleaner import (
    drop_missing_rows,
    fill_missing_with_mean,
    fill_missing_with_median,
    fill_missing_with_mode,
    get_cleaning_summary
)


from tools.visualization import (
    plot_histogram,
    plot_boxplot,
    plot_correlation_heatmap,
    plot_categorical_bar,
    plot_scatter_plotly
)


# ---------------------------------------
# 1. Page Configuration
# ---------------------------------------

st.set_page_config(
    page_title="DataInsight AI",
    page_icon="📊",
    layout="wide"
)



# ---------------------------------------
# 2. Session State Initialization
# ---------------------------------------

if "df" not in st.session_state:
    st.session_state.df = None

if "original_df" not in st.session_state:
    st.session_state.original_df = None

if "filename" not in st.session_state:
    st.session_state.filename = None

if "loaded_file_signature" not in st.session_state:
    st.session_state.loaded_file_signature = None

if "workflow_result" not in st.session_state:
    st.session_state.workflow_result = None

if "errors" not in st.session_state:
    st.session_state.errors = []

if "cleaning_preview" not in st.session_state:
    st.session_state.cleaning_preview = None

if "cleaning_method" not in st.session_state:
    st.session_state.cleaning_method = None

# ---------------------------------------
# 3. Application Header
# ---------------------------------------

st.title("DataInsight AI")

st.subheader(
    "Autonomous Multi-Agent Data Analysis Platform"
)

st.write(
    "Upload your dataset, explore its structure, "
    "and generate actionable business insights using AI."
)


# ---------------------------------------
# 4. Sidebar Navigation
# ---------------------------------------

with st.sidebar:

    st.title("DataInsight AI")

    st.write("Your AI-powered data analyst.")

    st.divider()

    st.subheader("Navigation")

    page = st.radio(
        "Select a page:",
        [
            "Dashboard",
            "Data Explorer",
            "AI Analysis",
            "Document Q&A"
        ]
    )

    st.divider()

    if st.session_state.df is not None:
        st.success("Dataset loaded successfully")
        st.caption(st.session_state.filename)
    else:
        st.info("No dataset uploaded yet.")

    st.divider()

    st.caption("Powered by Python, LangGraph and Gemini")


# ---------------------------------------
# 5. Dashboard Page
# ---------------------------------------

if page == "Dashboard":

    st.header("Welcome to DataInsight AI")

    st.write(
        "Start by uploading a CSV or Excel dataset "
        "to begin your analysis."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("AI Agents", "4")

    with col2:
        st.metric("Supported Formats", "CSV / Excel")

    with col3:
        st.metric("Analysis Engine", "LangGraph")

    if st.session_state.df is not None:

        st.divider()

        st.subheader("Current Dataset")

        df = st.session_state.df

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))

        st.caption(
            f"Loaded file: {st.session_state.filename}"
        )



# ---------------------------------------
# 6. Data Explorer Page
# ---------------------------------------

elif page == "Data Explorer":

    st.header("Data Explorer")

    st.write(
        "Upload, inspect and clean your dataset."
    )

    uploaded_file = st.file_uploader(
        "Choose a dataset",
        type=["csv", "xlsx", "xls"],
        help="Upload a CSV or Excel file."
    )

    if uploaded_file is not None:

        file_signature = (
            uploaded_file.name,
            uploaded_file.size
        )

        # Load only when a new file is selected
        if file_signature != st.session_state.loaded_file_signature:

            try:

                uploaded_file.seek(0)

                if uploaded_file.name.lower().endswith(".csv"):
                    df = pd.read_csv(uploaded_file)

                else:
                    df = pd.read_excel(uploaded_file)

                if df.empty:
                    st.error("The uploaded dataset is empty.")

                else:

                    st.session_state.original_df = df.copy()
                    st.session_state.df = df.copy()

                    st.session_state.filename = uploaded_file.name
                    st.session_state.loaded_file_signature = file_signature
                    st.session_state.workflow_result = None

                    st.success(
                        f"{uploaded_file.name} uploaded successfully!"
                    )

            except Exception as error:

                st.error(
                    f"Unable to read the uploaded file: {error}"
                )

    # Display dataset information
    if st.session_state.df is not None:

        df = st.session_state.df

        st.divider()

        # ---------------------------------------
        # Dataset Overview
        # ---------------------------------------

        st.subheader("Dataset Overview")

        overview = get_dataset_overview(df)

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Rows", overview["total_rows"])
        col2.metric("Total Columns", overview["total_columns"])

        col3.metric(
            "Memory Usage",
            f"{overview['memory_usage_kb']:.2f} KB"
        )

        # ---------------------------------------
        # Dataset Preview
        # ---------------------------------------

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        # ---------------------------------------
        # Column Information
        # ---------------------------------------

        st.subheader("Column Information")

        column_info = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Non-Null Values": df.notnull().sum().values,
            "Missing Values": df.isnull().sum().values
        })

        st.dataframe(
            column_info,
            use_container_width=True,
            hide_index=True
        )

        # ---------------------------------------
        # Missing Value Analysis
        # ---------------------------------------

        st.subheader("Missing Value Analysis")

        missing_values = get_missing_values(df)

        if missing_values.empty:

            st.success("No missing values found!")

        else:

            st.warning("Missing values detected.")

            st.dataframe(
                missing_values,
                use_container_width=True
            )

        
        # ----------------------------------
        # MISSING VALUE TREATMENT
        # ----------------------------------

        st.subheader("Missing Value Treatment")

        missing_columns = df.columns[df.isnull().any()].tolist()

        if not missing_columns:
            st.success("No missing values found in the current dataset!")

        else:
            st.write(
                "Choose a method to handle missing values "
                "and preview the results before applying changes."
            )

            cleaning_method = st.selectbox(
                "Select Cleaning Method",
                [
                    "Fill with Mean",
                    "Fill with Median",
                    "Fill with Mode",
                    "Drop Missing Rows"
                ],
                key="cleaning_method_select"
            )

            # Determine eligible columns
            numeric_missing_columns = [
                col for col in missing_columns
                if pd.api.types.is_numeric_dtype(df[col])
            ]

            if cleaning_method in ["Fill with Mean", "Fill with Median"]:
                available_columns = numeric_missing_columns
            else:
                available_columns = missing_columns

            if not available_columns:
                st.warning(
                    "No eligible columns found for this cleaning method."
                )

            else:
                selected_columns = st.multiselect(
                    "Select Columns",
                    options=available_columns,
                    default=available_columns,
                    key="cleaning_columns_select"
                )

                if st.button("Generate Cleaning Preview"):
                    try:
                        if not selected_columns:
                            st.warning("Please select at least one column.")

                        elif cleaning_method == "Fill with Mean":
                            preview_df = fill_missing_with_mean(
                                df, selected_columns
                            )

                        elif cleaning_method == "Fill with Median":
                            preview_df = fill_missing_with_median(
                                df, selected_columns
                            )

                        elif cleaning_method == "Fill with Mode":
                            preview_df = fill_missing_with_mode(
                                df, selected_columns
                            )

                        else:
                            preview_df = drop_missing_rows(
                                df, selected_columns
                            )

                        if selected_columns:
                            st.session_state.cleaning_preview = {
                                "df": preview_df,
                                "method": cleaning_method,
                                "columns": selected_columns.copy()
                            }

                    except Exception as e:
                        st.error(f"Cleaning preview failed: {e}")

            # Display cleaning preview
            preview_data = st.session_state.cleaning_preview

            if preview_data is not None:
                preview_df = preview_data["df"]

                st.divider()
                st.subheader("Cleaning Preview")

                st.write(f"**Method:** {preview_data['method']}")
                st.write(
                    f"**Selected columns:** "
                    f"{', '.join(preview_data['columns'])}"
                )

                # Compare before and after
                summary = get_cleaning_summary(df, preview_df)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Original Rows",
                        summary["original_rows"]
                    )

                with col2:
                    st.metric(
                        "Cleaned Rows",
                        summary["cleaned_rows"]
                    )

                with col3:
                    st.metric(
                        "Missing Values Resolved",
                        summary["missing_values_resolved"]
                    )

                st.write("**Preview of Cleaned Dataset**")
                st.dataframe(preview_df.head(10), use_container_width=True)

                # Check whether the preview matches current selections
                preview_is_current = (
                    preview_data["method"] == cleaning_method
                    and preview_data["columns"] == selected_columns
                )

                if not preview_is_current:
                    st.warning(
                        "Your cleaning settings have changed. "
                        "Generate a new preview before applying."
                    )

                else:
                    if st.button("Apply Cleaning Changes"):
                        st.session_state.df = preview_df.copy()
                        st.session_state.workflow_result = None
                        st.session_state.cleaning_preview = None

                        st.success("Cleaning changes applied successfully!")
                        st.rerun()


        # ---------------------------------------
        # Duplicate Record Analysis
        # ---------------------------------------

        st.subheader("Duplicate Record Analysis")

        duplicate_report = get_duplicate_records(df)

        duplicate_count = duplicate_report[
            "total_duplicate_rows"
        ]

        duplicate_percentage = (
            duplicate_count / len(df)
        ) * 100

        col1, col2 = st.columns(2)

        col1.metric(
            "Duplicate Records",
            duplicate_count
        )

        col2.metric(
            "Duplicate Percentage",
            f"{duplicate_percentage:.2f}%"
        )

        if duplicate_count > 0:

            st.warning(
                f"Found {duplicate_count} duplicate rows "
                "that can be removed."
            )

            with st.expander("View Duplicate Records"):

                st.dataframe(
                    pd.DataFrame(
                        duplicate_report["duplicate_records"]
                    ),
                    use_container_width=True
                )

            if st.button(
                "Remove Duplicate Records",
                type="primary"
            ):

                cleaned_df = df.drop_duplicates(
                    keep="first"
                ).copy()

                removed_count = len(df) - len(cleaned_df)

                st.session_state.df = cleaned_df
                st.session_state.workflow_result = None

                st.success(
                    f"Successfully removed {removed_count} "
                    "duplicate row(s)."
                )

                st.rerun()

        else:

            st.success("No duplicate records found!")

        # ---------------------------------------
        # Restore Original Dataset
        # ---------------------------------------

        if st.button("Restore Original Dataset"):

            st.session_state.df = (
                st.session_state.original_df.copy()
            )

            st.session_state.workflow_result = None

            st.success("Original dataset restored.")

            st.rerun()

        # ---------------------------------------
        # Remove Current Dataset
        # ---------------------------------------

        if st.button("Remove Current Dataset"):

            st.session_state.df = None
            st.session_state.original_df = None
            st.session_state.filename = None
            st.session_state.loaded_file_signature = None
            st.session_state.workflow_result = None

            st.rerun()

        st.divider()


        # ----------------------------------
        # DATA VISUALIZATION
        # ----------------------------------

        st.divider()
        st.header("Data Visualization")

        st.write(
            "Generate interactive and statistical visualizations "
            "from your current dataset."
        )

        chart_type = st.selectbox(
            "Select Visualization Type",
            [
                "Histogram",
                "Box Plot",
                "Correlation Heatmap",
                "Bar Chart",
                "Scatter Plot"
            ],
            key="chart_type"
        )

        numeric_columns = df.select_dtypes(include="number").columns.tolist()

        categorical_columns = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()


        if chart_type == "Histogram":

            if numeric_columns:
                selected_column = st.selectbox(
                    "Select Numerical Column",
                    numeric_columns,
                    key="histogram_column"
                )

                bins = st.slider(
                    "Number of Bins",
                    min_value=5,
                    max_value=30,
                    value=10
                )

                if st.button("Generate Histogram"):
                    try:
                        fig = plot_histogram(
                            df, selected_column, bins=bins
                        )
                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Unable to generate histogram: {e}")

            else:
                st.warning("No numerical columns available.")


        elif chart_type == "Box Plot":

            if numeric_columns:
                selected_column = st.selectbox(
                    "Select Numerical Column",
                    numeric_columns,
                    key="boxplot_column"
                )

                if st.button("Generate Box Plot"):
                    try:
                        fig = plot_boxplot(df, selected_column)
                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Unable to generate box plot: {e}")

            else:
                st.warning("No numerical columns available.")


        elif chart_type == "Correlation Heatmap":

            if len(numeric_columns) >= 2:

                if st.button("Generate Correlation Heatmap"):
                    try:
                        fig = plot_correlation_heatmap(df)
                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Unable to generate heatmap: {e}")

            else:
                st.warning(
                    "At least two numerical columns are required."
                )


        elif chart_type == "Bar Chart":

            if categorical_columns:
                selected_column = st.selectbox(
                    "Select Categorical Column",
                    categorical_columns,
                    key="bar_column"
                )

                if st.button("Generate Bar Chart"):
                    try:
                        fig = plot_categorical_bar(
                            df, selected_column
                        )
                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Unable to generate bar chart: {e}")

            else:
                st.warning("No categorical columns available.")


        elif chart_type == "Scatter Plot":

            if len(numeric_columns) >= 2:

                col1, col2 = st.columns(2)

                with col1:
                    x_column = st.selectbox(
                        "Select X-axis",
                        numeric_columns,
                        key="scatter_x"
                    )

                with col2:
                    y_column = st.selectbox(
                        "Select Y-axis",
                        numeric_columns,
                        index=1,
                        key="scatter_y"
                    )

                color_options = ["None"] + categorical_columns

                color_column = st.selectbox(
                    "Color By (Optional)",
                    color_options,
                    key="scatter_color"
                )

                if st.button("Generate Scatter Plot"):
                    try:
                        fig = plot_scatter_plotly(
                            df,
                            x_column,
                            y_column,
                            None if color_column == "None" else color_column
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                    except Exception as e:
                        st.error(f"Unable to generate scatter plot: {e}")

            else:
                st.warning(
                    "At least two numerical columns are required."
                )

        st.divider()


# ---------------------------------------
# 7. AI Analysis Page
# ---------------------------------------

elif page == "AI Analysis":

    st.header("AI-Powered Data Analysis")

    st.write(
        "Ask questions about your dataset "
        "and get clear, AI-powered answers."
    )

    if st.session_state.df is None:

        st.warning(
            "Please upload a dataset in the Data Explorer first."
        )

    else:

        df = st.session_state.df

        st.success(
            f"Dataset ready: {st.session_state.filename}"
        )

        st.divider()

        # ---------------------------------------
        # Natural Language Question
        # ---------------------------------------

        st.subheader("Ask Your Data")

        user_request = st.text_area(
            "Enter your analysis question:",
            placeholder=(
                "Example: Which product category generates "
                "the highest average purchase amount?"
            ),
            height=100
        )

        # ---------------------------------------
        # Optional Detailed Report
        # ---------------------------------------

        include_detailed_report = st.checkbox(
            "Include Detailed Business Report",
            value=False,
            help=(
                "Enable this option to generate detailed business "
                "insights, recommendations and data limitations. "
                "This requires an additional Gemini API call."
            )
        )

        # ---------------------------------------
        # Execute Analysis
        # ---------------------------------------

        if st.button(
            "Run AI Analysis",
            type="primary",
            use_container_width=True
        ):

            if not user_request.strip():

                st.warning(
                    "Please enter a question before running analysis."
                )

            else:

                # Clear previous result
                st.session_state.workflow_result = None

                try:

                    with st.spinner(
                        "AI Agents are analyzing your dataset..."
                    ):

                        result = run_workflow(
                            df,
                            user_request,
                            include_detailed_report=include_detailed_report
                        )

                        st.session_state.workflow_result = result

                    if result.get("direct_answer"):

                        st.success(
                            "AI Analysis completed successfully!"
                        )

                    else:

                        st.warning(
                            "The workflow could not generate a quick answer."
                        )

                except Exception as error:

                    st.error(
                        f"Analysis failed: {str(error)}"
                    )

        # ---------------------------------------
        # Display Analysis Results
        # ---------------------------------------

        result = st.session_state.workflow_result

        if result is not None:

            st.divider()

            # ---------------------------------------
            # Workflow Errors
            # ---------------------------------------

            workflow_errors = result.get("errors", [])

            if workflow_errors:

                st.warning("Some workflow errors occurred.")

                for error in workflow_errors:
                    st.write(f"- {error}")

            # ---------------------------------------
            # Quick Answer (Always Displayed)
            # ---------------------------------------

            direct_answer = result.get("direct_answer", "")

            if direct_answer:

                st.subheader("Quick Answer")

                st.success(direct_answer)

            else:

                st.error(
                    "A direct answer could not be generated."
                )

            # ---------------------------------------
            # Optional Detailed Business Report
            # ---------------------------------------

            if result.get("include_detailed_report", False):

                insight_report = result.get(
                    "insight_report", {}
                )

                if insight_report:

                    st.divider()

                    st.subheader("Detailed Business Report")

                    st.caption(
                        f"Generated using: "
                        f"{insight_report.get('writer_mode', 'unknown')}"
                    )

                    # Business Insights
                    st.subheader("Business Insights")

                    insights = insight_report.get(
                        "insights", []
                    )

                    if insights:

                        for index, insight in enumerate(
                            insights, start=1
                        ):

                            st.markdown(
                                f"**{index}.** {insight}"
                            )

                    else:

                        st.info("No business insights generated.")

                    # Recommendations
                    st.subheader("Recommendations")

                    recommendations = insight_report.get(
                        "recommendations", []
                    )

                    if recommendations:

                        for index, recommendation in enumerate(
                            recommendations, start=1
                        ):

                            st.markdown(
                                f"**{index}.** {recommendation}"
                            )

                    else:

                        st.info("No recommendations generated.")

                    # Data Limitations
                    st.subheader("Data Limitations")

                    limitations = insight_report.get(
                        "data_limitations", []
                    )

                    if limitations:

                        for limitation in limitations:

                            st.markdown(f"- {limitation}")

                    else:

                        st.info("No limitations reported.")

                else:

                    st.warning(
                        "The Quick Answer was generated, "
                        "but the detailed report could not be generated."
                    )

            else:

                st.info(
                    "Detailed report was not requested. "
                    "Enable the checkbox before running another query "
                    "if you want business insights and recommendations."
                )

            # ---------------------------------------
            # Technical Execution Details
            # ---------------------------------------

            with st.expander("View Analysis Execution Details"):

                st.write("Analysis Plan")

                st.json(
                    result.get("analysis_plan", {})
                )

                st.write("Analysis Results")

                st.json(
                    result.get("analysis_results", {})
                )

                st.write("Workflow Errors")

                st.write(
                    result.get("errors", [])
                )


# ---------------------------------------
# 8. PDF Document Q&A
# ---------------------------------------

elif page == "Document Q&A":

    st.header("PDF Document Q&A")
    st.write(
        "Upload a text-based PDF and ask questions about its contents. "
        "Answers include page references."
    )

    uploaded_pdf = st.file_uploader(
        "Choose a PDF document",
        type=["pdf"],
        help="Text-based PDFs are supported. Scanned PDFs requiring OCR are not supported yet."
    )

    if uploaded_pdf is not None:
        pdf_bytes = uploaded_pdf.getvalue()
        pdf_signature = hashlib.sha256(pdf_bytes).hexdigest()

        if st.session_state.get("pdf_signature") != pdf_signature:
            try:
                with st.spinner("Extracting text from PDF pages..."):
                    extracted = extract_pdf_text(pdf_bytes)

                st.session_state.pdf_signature = pdf_signature
                st.session_state.pdf_document = extracted
                st.session_state.pdf_filename = uploaded_pdf.name
                st.session_state.pdf_answer = ""

            except Exception as error:
                st.session_state.pdf_signature = None
                st.session_state.pdf_document = None
                st.error(f"PDF processing failed: {error}")

    document = st.session_state.get("pdf_document")

    if document is not None:
        st.success(f"Document ready: {st.session_state.pdf_filename}")

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Total PDF pages", document["total_pages"])
        metric2.metric("Pages with text", document["pages_with_text"])
        metric3.metric("Characters extracted", f'{document["characters_extracted"]:,}')

        if document["pages_without_text"]:
            st.warning(
                "No selectable text was extracted from page(s): "
                + ", ".join(map(str, document["pages_without_text"]))
                + ". These may be scanned pages or image-only pages."
            )

        if document["truncated"]:
            st.warning(
                "The extraction reached the current 60,000-character limit. "
                "Some later document text may not be included in answers."
            )

        with st.expander("Preview extracted text"):
            for page_data in document["pages"][:5]:
                st.markdown(f"**Page {page_data['page']}**")
                st.write(page_data["text"][:2500])

        st.divider()
        question = st.text_area(
            "Ask a question about this document",
            placeholder="Example: What are the main objectives described in the document?",
            height=100,
            key="pdf_question"
        )

        if st.button("Ask PDF", type="primary", use_container_width=True):
            if not question.strip():
                st.warning("Enter a question first.")
            else:
                try:
                    with st.spinner("Reading the document and preparing an answer..."):
                        answer = generate_document_answer(question, document)
                    st.session_state.pdf_answer = answer
                except Exception as error:
                    st.error(f"Could not answer the question: {error}")

        if st.session_state.get("pdf_answer"):
            st.subheader("Document Answer")
            st.markdown(st.session_state.pdf_answer)

    else:
        st.info("Upload a PDF to begin document analysis.")
