def words2characters(words):
    """
    This function converts a list of words into a list of characters.

    @param:
    words - a list of words

    @return:
    characters - a list of characters
    """
    characters = []

    for word in words:
        for char in str(word):
            characters.append(char)

    return characters