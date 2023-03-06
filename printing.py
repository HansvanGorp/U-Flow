def print_table(table_name, row_names, headers, values, lower = True): 
    # create header
    no_methods = len(headers)
    top = table_name.center(19)+'|'
    for method_id in range(no_methods):
        top += headers[method_id].center(12)
        top += '|'
    
    print('\n\n')
    print(top)
    print('-------------------'+no_methods*'-------------'+'-')
        
    # fill in table
    for row_id,row_name in enumerate(row_names): 
        print(f"{row_name:<14}",end="")
        for method_id in range(no_methods):
            to_print = values[row_id,method_id,:].mean()
            print(f"| {to_print:9.3f}  ",end="")    
        print("|")