import GUI from 'lil-gui';
import type { IdmParams } from '../sim/idm';

export interface TrafficSettings {
  timeScale: number;
  carLength: number;
  cars: [IdmParams, IdmParams];
}

export interface GuiState {
  settings: TrafficSettings;
  telemetry: { carA: string; carB: string };
}

/** Creates the settings panel and returns its live values. */
export function setupGui(): GuiState {
  const state: GuiState = {
    settings: {
      timeScale: 1,
      carLength: 4.5,
      cars: [
        { v0: 30, T: 1.5, a: 2.0, b: 2.5, s0: 2, delta: 4 },
        { v0: 12, T: 1.5, a: 1.2, b: 2.0, s0: 2, delta: 4 },
      ],
    },
    telemetry: { carA: '', carB: '' },
  };

  const gui = new GUI({ title: 'Traffic' });
  gui.add(state.settings, 'timeScale', 0.1, 3, 0.1).name('Time scale');
  gui.add(state.settings, 'carLength', 3, 8, 0.5).name('Vehicle length (m)');

  const names = ['Car A (red)', 'Car B (blue)'];
  state.settings.cars.forEach((car, i) => {
    const folder = gui.addFolder(names[i]);
    folder.add(car, 'v0', 5, 40, 1).name('Desired speed (m/s)');
    folder.add(car, 'a', 0.5, 4, 0.1).name('Max accel (m/s²)');
    folder.add(car, 'b', 0.5, 6, 0.1).name('Comfy decel (m/s²)');
    folder.add(car, 'T', 0.5, 3, 0.1).name('Time headway (s)');
    folder.add(car, 's0', 0.5, 10, 0.5).name('Min distance (m)');
    folder.add(car, 'delta', 1, 10, 1).name('Accel exponent');
  });

  gui.add(state.telemetry, 'carA').name('A speed').listen();
  gui.add(state.telemetry, 'carB').name('B speed').listen();
  return state;
}
