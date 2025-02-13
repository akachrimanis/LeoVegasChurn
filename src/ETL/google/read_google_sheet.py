import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials


def read_google_sheet(
    spreadsheet_id: str, worksheet_name: str, credentials_json: str, scope: list = None
) -> pd.DataFrame:
    """
    Read data from a specific Google Sheet worksheet and return as a DataFrame.

    Args:
        spreadsheet_id (str): The ID of the Google Sheets document.
        worksheet_name (str): The name of the worksheet to extract data from.
        credentials_json (str): Path to the Google service account credentials in JSON format.
        scope (list, optional): The scope of the authorization. Default is None.

    Returns:
        pd.DataFrame: DataFrame containing the extracted data.
    """
    try:
        # Set default scope if not provided
        if scope is None:
            scope = [
                "https://spreadsheets.google.com/feeds",
                "https://www.googleapis.com/auth/drive",
            ]

        # Setup credentials and authorize the client
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            credentials_json, scope
        )
        client = gspread.authorize(creds)

        # Open the Google Sheet and select the worksheet
        sheet = client.open_by_key(spreadsheet_id)
        worksheet = sheet.worksheet(worksheet_name)

        # Get data from the worksheet
        data = pd.DataFrame(worksheet.get_all_records())
        print(
            f"Data extracted successfully from Google Sheet: {spreadsheet_id}, Worksheet: {worksheet_name}"
        )
        return data

    except gspread.exceptions.APIError as api_error:
        print(f"API Error when accessing Google Sheet: {api_error}")
        raise
    except Exception as e:
        print(f"Error reading Google Sheet {spreadsheet_id}/{worksheet_name}: {e}")
        raise
