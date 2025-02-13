import pandas as pd
from datetime import datetime, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar


def create_extended_date_features(data: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Creates extended date-related features from a specified date column in the DataFrame.

    Args:
        data (pd.DataFrame): Input DataFrame.
        date_column (str): Name of the column containing date values.

    Returns:
        pd.DataFrame: DataFrame with new date features added.
    """
    try:
        # Ensure the date column is in datetime format
        data[date_column] = pd.to_datetime(
            data[date_column], format="%Y-%m-%d", errors="raise"
        )

        # Drop rows where date parsing failed
        data = data.dropna(subset=[date_column])

        # Extract basic date features
        data[f"{date_column}_year"] = data[date_column].dt.year.astype(str)
        data[f"{date_column}_month_name"] = data[date_column].dt.strftime(
            "%B"
        )  # Add month name
        data[f"{date_column}_day"] = data[date_column].dt.day.astype(str)
        data[f"{date_column}_weekday"] = data[date_column].dt.weekday
        data[f"{date_column}_week"] = (
            data[date_column].dt.isocalendar().week.astype(str)
        )
        data[f"{date_column}_quarter"] = data[date_column].dt.quarter
        data[f"{date_column}_is_weekend"] = data[date_column].dt.weekday >= 5
        data[f"{date_column}_is_month_start"] = data[date_column].dt.is_month_start
        data[f"{date_column}_is_month_end"] = data[date_column].dt.is_month_end
        data[f"{date_column}_dayofyear"] = data[date_column].dt.day_of_year

        # Add days until Christmas
        data[f"{date_column}_days_until_christmas"] = (
            pd.to_datetime(data[date_column].dt.year.astype(str) + "-12-25")
            - data[date_column]
        ).dt.days.clip(lower=0)

        # Add days until Easter (using a simple method)
        def calculate_easter(year):
            """Calculate Easter date using the Anonymous Gregorian algorithm."""
            a = year % 19
            b = year // 100
            c = year % 100
            d = b // 4
            e = b % 4
            f = (b + 8) // 25
            g = (b - f + 1) // 3
            h = (19 * a + b - d - g + 15) % 30
            i = c // 4
            k = c % 4
            l = (32 + 2 * e + 2 * i - h - k) % 7
            m = (a + 11 * h + 22 * l) // 451
            month = (h + l - 7 * m + 114) // 31
            day = ((h + l - 7 * m + 114) % 31) + 1
            return datetime(year, month, day)

        easter_dates = {
            year: calculate_easter(year) for year in data[date_column].dt.year.unique()
        }
        data[f"{date_column}_days_until_easter"] = (
            data[date_column]
            .apply(
                lambda x: (easter_dates[x.year] - x).days
                if x.year in easter_dates
                else None
            )
            .clip(lower=0)
        )

        # Add days until Black Friday (4th Friday of November)
        def calculate_black_friday(year):
            """Calculate Black Friday (4th Friday of November)."""
            first_day_of_november = datetime(year, 11, 1)
            weekday_of_first = first_day_of_november.weekday()
            days_until_friday = (4 - weekday_of_first) % 7
            return first_day_of_november + timedelta(days=days_until_friday + 21)

        black_friday_dates = {
            year: calculate_black_friday(year)
            for year in data[date_column].dt.year.unique()
        }
        data[f"{date_column}_days_until_black_friday"] = (
            data[date_column]
            .apply(
                lambda x: (black_friday_dates[x.year] - x).days
                if x.year in black_friday_dates
                else None
            )
            .clip(lower=0)
        )

        # Add vacation period category
        def vacation_period(date):
            """Categorize date into common vacation periods."""
            month_day = (date.month, date.day)
            if (6, 1) <= month_day <= (8, 31):  # Summer vacation (June to August)
                return "Summer Vacation"
            elif (12, 15) <= month_day <= (12, 31) or (1, 1) <= month_day <= (
                1,
                15,
            ):  # Christmas/New Year
                return "Winter Vacation"
            elif (3, 15) <= month_day <= (4, 15):  # Spring Break
                return "Spring Break"
            else:
                return "No Vacation"

        data[f"{date_column}_vacation_period"] = data[date_column].apply(
            vacation_period
        )

        return data

    except Exception as e:
        print(f"Error while creating extended date features: {e}")
        return data
