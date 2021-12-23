def grid(start, distance, rows, columns, z = None):
  coords = []
  for row in range(0, rows):
    for column in range(0, columns):
      x = start[0] + column * distance[0]
      y = start[1] + row * distance[1]
      coordinate = [x,y]

      if (z is not None):
        coordinate.append(z)
      coords.append(coordinate)
  return coords