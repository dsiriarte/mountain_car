| Método | Semilla | Pasos de entorno | 1.ª bandera (episodio) | Media móvil ≥ −150 (episodio) | Media últimos 100 (entrenamiento) | Evaluación 100 ep. (media ± desv.) | Mejor / peor | Llega a la bandera |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q-Learning tabular | 0 | 3216885 | 1903 | 7439 | -157.1 | -165.1 ± 14.3 | -124 / -200 | 90/100 |
| Q-Learning tabular | 1 | 3227742 | 1538 | 7827 | -134.0 | -133.8 ± 12.9 | -118 / -155 | 100/100 |
| Q-Learning tabular | 2 | 3272494 | 1741 | 5875 | -132.5 | -133.3 ± 17.9 | -108 / -168 | 100/100 |
| Q-Learning tabular | 3 | 3279748 | 1725 | 6503 | -150.2 | -164.7 ± 23.2 | -121 / -199 | 100/100 |
| Q-Learning tabular | 4 | 3241981 | 1613 | 7620 | -151.2 | -141.0 ± 7.9 | -115 / -152 | 100/100 |
| Q-Learning tabular | 5 | 3208365 | 1765 | 6575 | -164.8 | -164.6 ± 12.2 | -131 / -181 | 100/100 |
| Q-Learning tabular **(mejor)** | 6 | 3147729 | 1654 | 6271 | -127.3 | -128.3 ± 15.1 | -111 / -151 | 100/100 |
| Q-Learning tabular | 7 | 3194457 | 1536 | 6359 | -160.7 | -182.0 ± 12.9 | -148 / -200 | 82/100 |
| Q-Learning tabular | 8 | 3274881 | 1998 | 7291 | -161.1 | -157.2 ± 8.8 | -145 / -171 | 100/100 |
| Q-Learning tabular | 9 | 3239897 | 1786 | 7729 | -132.7 | -130.2 ± 18.0 | -108 / -155 | 100/100 |
| DQN | 0 | 361357 | 10 | 1002 | -109.1 | -110.1 ± 13.3 | -86 / -140 | 100/100 |
| DQN | 1 | 500000 | nunca | nunca | -200.0 | -200.0 ± 0.0 | -200 / -200 | 0/100 |
| DQN | 2 | 364507 | 5 | 914 | -127.1 | -122.0 ± 25.3 | -84 / -184 | 100/100 |
| DQN | 3 | 500000 | nunca | nunca | -200.0 | -200.0 ± 0.0 | -200 / -200 | 0/100 |
| DQN | 4 | 345876 | 1 | 829 | -115.0 | -104.7 ± 7.5 | -84 / -114 | 100/100 |
| DQN | 5 | 356693 | 1 | 854 | -106.9 | -104.7 ± 7.6 | -84 / -113 | 100/100 |
| DQN **(mejor)** | 6 | 336573 | 18 | 808 | -106.9 | -103.4 ± 6.8 | -84 / -110 | 100/100 |
| DQN | 7 | 348054 | 23 | 857 | -113.8 | -113.6 ± 21.4 | -84 / -184 | 100/100 |
| DQN | 8 | 384102 | 6 | 867 | -121.4 | -107.8 ± 19.8 | -83 / -200 | 98/100 |
| DQN | 9 | 349185 | 4 | 841 | -115.4 | -131.7 ± 29.2 | -97 / -200 | 86/100 |

**Q-Learning tabular**: 8/10 semillas llegan a la bandera en los 100 episodios de evaluación y 0/10 nunca llegan; evaluación media -150.0 (mediana -149.1, rango -182.0 a -128.3); media de las semillas que llegan siempre: -144.1.

**DQN**: 6/10 semillas llegan a la bandera en los 100 episodios de evaluación y 2/10 nunca llegan; evaluación media -129.8 (mediana -111.9, rango -200.0 a -103.4); media de las semillas que llegan siempre: -109.8.

**Confirmación del mejor Q-Learning tabular (semilla 6)** en 100 estados iniciales nuevos: -127.8 ± 14.8, mejor -111, peor -153, banderas 100/100.

**Confirmación del mejor DQN (semilla 6)** en 100 estados iniciales nuevos: -100.4 ± 8.3, mejor -83, peor -110, banderas 100/100.
