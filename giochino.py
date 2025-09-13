

import math
import sys
import pygame


WIDTH, HEIGHT = 800, 600
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LINE_COLOR = (0, 200, 255)
FPS = 60

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Cubo 3D Rotante - Pygame")
clock = pygame.time.Clock()


scale = 150
vertices = [
    (-1, -1, -1),
    (-1, -1,  1),
    (-1,  1, -1),
    (-1,  1,  1),
    ( 1, -1, -1),
    ( 1, -1,  1),
    ( 1,  1, -1),
    ( 1,  1,  1),
]


edges = [
    (0, 1), (0, 2), (0, 4),
    (3, 1), (3, 2), (3, 7),
    (5, 1), (5, 4), (5, 7),
    (6, 2), (6, 4), (6, 7)
]


distance = 5.0
z_offset = 5.0  # trasla il cubo in avanti (lontano dalla camera)

def rotate_x(x, y, z, angle):
    """Rotazione intorno all'asse X."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    y2 = y * cos_a - z * sin_a
    z2 = y * sin_a + z * cos_a
    return x, y2, z2

def rotate_y(x, y, z, angle):
    """Rotazione intorno all'asse Y."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x2 = x * cos_a + z * sin_a
    z2 = -x * sin_a + z * cos_a
    return x2, y, z2

def rotate_z(x, y, z, angle):
    """Rotazione intorno all'asse Z."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x2 = x * cos_a - y * sin_a
    y2 = x * sin_a + y * cos_a
    return x2, y2, z

def project(x, y, z):
    """Proiezione prospettica 3D -> 2D."""
    # Evita divisione per zero se z è troppo vicino a -distance
    denom = (z + distance)
    if denom == 0:
        denom = 1e-6
    factor = 1.0 / denom
    x_proj = x * factor
    y_proj = y * factor
    # Trasforma in coordinate schermo
    sx = int(WIDTH / 2 + x_proj * scale)
    sy = int(HEIGHT / 2 - y_proj * scale)
    return sx, sy

def main():
    angle_x = 0.0
    angle_y = 0.0
    angle_z = 0.0
    running = True
    paused = False

    while True:
        dt = clock.tick(FPS) / 1000.0  # secondi per frame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused

        if not paused:
            # Velocità di rotazione (radianti al secondo)
            angle_x += 0.7 * dt
            angle_y += 1.0 * dt
            angle_z += 0.5 * dt

        screen.fill(BLACK)

        # Calcola i vertici trasformati (rotazione + traslazione lungo z)
        transformed = []
        for (x, y, z) in vertices:
            # Rotazioni
            x1, y1, z1 = rotate_x(x, y, z, angle_x)
            x2, y2, z2 = rotate_y(x1, y1, z1, angle_y)
            x3, y3, z3 = rotate_z(x2, y2, z2, angle_z)

            # Traslazione in avanti per essere davanti alla camera
            z3 += z_offset

            # Proiezione prospettica
            sx, sy = project(x3, y3, z3)
            transformed.append((sx, sy))

        # Disegna gli spigoli
        for a, b in edges:
            pygame.draw.line(screen, LINE_COLOR, transformed[a], transformed[b], 2)

        # Informazioni a schermo
        font = pygame.font.SysFont(None, 24)
        info = font.render("ESC per uscire | SPAZIO per pausa", True, WHITE)
        screen.blit(info, (10, 10))

        pygame.display.flip()

    #pygame.quit()
    #sys.exit(0)

if __name__ == "main":
    main()