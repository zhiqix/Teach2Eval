def f(items):
    for item in items:
        item[1] =1
    
items = [{},{}]

f(items)
print(items)