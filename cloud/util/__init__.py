def get_text(nodelist):
    '''
    This helper method extracts text from a node object in a xml tree.
    '''
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return ''.join(rc)

def get_float_metadata(tree, tag):
    '''
    Gets and parses the text inside a a node with a particular tag into a float.
    '''
    return float(get_metadata(tree, tag))

def get_metadata(tree, tag):
    '''
    Gets and parses the text inside a node.
    '''
    elements = tree.getElementsByTagName(tag)
    text = get_text(elements[0].childNodes)
    return text