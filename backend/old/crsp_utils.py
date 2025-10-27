def get_tickers_from_permnos(db, permnos):
    """Retrieves the ticker symbols corresponding to a list of permnos from the CRSP database.

    This function queries the 'stocknames' table in the CRSP database to get the ticker symbols
    that correspond to each permno in the input list. The resulting ticker symbols are returned in 
    the same order as the permnos in the input list.

    Args:
        db (wrds.Connection): An open WRDS database connection object. This is used to perform the SQL query.
        permnos (list of int): A list of permnos for which to retrieve corresponding ticker symbols.

    Returns:
        list of str: The ticker symbols corresponding to the input permnos, returned in the same order as the input.

    Raises:
        KeyError: If a permno in the input list does not have a corresponding ticker in the 'stocknames' table.
    """
    permno_list = ', '.join([str(permno) for permno in permnos])
    query = f"""
    SELECT sn.ticker, sn.permno
    FROM crsp.stocknames AS sn
    WHERE sn.permno IN ({permno_list})
    """
    data = db.raw_sql(query)
    
    # Create a dictionary mapping permno to ticker
    permno_to_ticker = dict(zip(data.permno, data.ticker))
    
    # Return the list of tickers corresponding to the input permnos
    tickers = [permno_to_ticker[permno] for permno in permnos]
    return tickers
