def next_birthday(date, birthdays):
    '''
    Find the next birthday after the given date.

    @param:
    date - a tuple of two integers specifying (month, day)
    birthdays - a dict mapping from date tuples to lists of names

    @return:
    birthday - the next day, after given date, on which somebody has a birthday
    list_of_names - list of all people with birthdays on that date
    '''
    
    sorted_dates = sorted(birthdays.keys())

    for birthday in sorted_dates:
        if birthday > date:
            return birthday, birthdays[birthday]

    # Wrap around to the first birthday of the next year
    birthday = sorted_dates[0]
    return birthday, birthdays[birthday]