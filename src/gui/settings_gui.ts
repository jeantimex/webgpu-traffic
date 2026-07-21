import GUI, { Controller } from 'lil-gui';
import type { IdmParams } from '../sim/idm';
import type { RoadConfig } from '../sim/road';

export interface TrafficSettings {
  scene: number;
  timeScale: number;
  carLength: number;
  dayMode: boolean;
  light: { green: number; yellow: number; red: number; override: 'auto' | 'green' | 'yellow' | 'red' };
  cars: IdmParams[];
  scene3: RoadConfig;
  scene3B: RoadConfig;
}

export interface GuiState {
  settings: TrafficSettings;
  telemetry: { speeds: Record<string, string>; light: string };
  selectedCar: number;
  /** Set by the renderer every frame: false when no safe spawn slot exists. */
  canSpawn: boolean;
}

/** Default IDM params for an added car (copied on add). */
export const NEW_CAR_PARAMS: IdmParams = { v0: 20, T: 1.5, a: 2.0, b: 2.5, s0: 2, delta: 4 };

/** Hard cap on cars (the renderer preallocates this many uniform slots). */
export const MAX_CARS = 16;

export const carName = (i: number): string => `Car ${i + 1}`;

/** Creates the settings panel and returns its live values. */
export function setupGui(): GuiState {
  const state: GuiState = {
    settings: {
      scene: 1,
      timeScale: 1,
      carLength: 4.5,
      dayMode: true,
      light: { green: 10, yellow: 2, red: 8, override: 'auto' },
      cars: [
        { v0: 30, T: 1.5, a: 2.0, b: 2.5, s0: 2, delta: 4 },
        { v0: 12, T: 1.5, a: 1.2, b: 2.0, s0: 2, delta: 4 },
        { v0: 20, T: 1.5, a: 1.8, b: 2.2, s0: 2, delta: 4 },
        { v0: 25, T: 1.5, a: 2.2, b: 2.8, s0: 2, delta: 4 },
      ],
      scene3: { shape: 'straight', length: 150, radius: 50, angle: 90, lanesForward: 2, lanesBackward: 0 },
      scene3B: { shape: 'arc', length: 150, radius: 50, angle: 90, lanesForward: 2, lanesBackward: 0 },
    },
    telemetry: { speeds: {}, light: '' },
    selectedCar: 0,
    canSpawn: true,
  };

  const gui = new GUI({ title: 'Traffic' });
  gui
    .add(state.settings, 'scene', {
      'Scene 1 (ring)': 1,
      'Scene 2 (square)': 2,
      'Scene 3 (road)': 3,
    })
    .name('Scene');
  gui.add(state.settings, 'timeScale', 0.1, 3, 0.1).name('Time scale');
  gui.add(state.settings, 'carLength', 3, 8, 0.5).name('Vehicle length (m)');
  gui.add(state.settings, 'dayMode').name('Daylight');

  const addRoadFolder = (title: string, config: RoadConfig): void => {
    const folder = gui.addFolder(title);
    const shapeController = folder
      .add(config, 'shape', { Straight: 'straight', Arc: 'arc', 'S-curve': 'scurve' })
      .name('Shape');
    const lengthController = folder.add(config, 'length', 50, 400, 10).name('Length (m)');
    const radiusController = folder.add(config, 'radius', 20, 100, 5).name('Radius (m)');
    const angleController = folder.add(config, 'angle', -180, 180, 5).name('Angle (°)');
    // Only the parameters of the selected shape are shown (length for straight, radius/angle for curves).
    const updateVisibility = (): void => {
      const straight = config.shape === 'straight';
      lengthController.show(straight);
      radiusController.show(!straight);
      angleController.show(!straight);
    };
    shapeController.onChange(updateVisibility);
    updateVisibility();
    folder.add(config, 'lanesForward', 1, 3, 1).name('Lanes forward');
    folder.add(config, 'lanesBackward', 0, 3, 1).name('Lanes back (0=one-way)');
  };
  addRoadFolder('Road A (scene 3)', state.settings.scene3);
  addRoadFolder('Road B (scene 3)', state.settings.scene3B);

  const lightFolder = gui.addFolder('Traffic light');
  lightFolder
    .add(state.settings.light, 'override', { Auto: 'auto', Green: 'green', Yellow: 'yellow', Red: 'red' })
    .name('Mode');
  lightFolder.add(state.settings.light, 'green', 2, 60, 1).name('Green (s)');
  lightFolder.add(state.settings.light, 'yellow', 1, 10, 0.5).name('Yellow (s)');
  lightFolder.add(state.settings.light, 'red', 2, 60, 1).name('Red (s)');

  const carFolder = gui.addFolder('Car');
  const paramControllers: Controller[] = [];

  /** Rebuilds the dropdown, the selected car's sliders, and the speed readouts. */
  const refreshCarPanel = (): void => {
    state.selectedCar = Math.min(state.selectedCar, state.settings.cars.length - 1);
    listController.options(Object.fromEntries(state.settings.cars.map((_, i) => [carName(i), i])));

    paramControllers.forEach((c) => c.destroy());
    paramControllers.length = 0;
    const car = state.settings.cars[state.selectedCar];
    paramControllers.push(
      carFolder.add(car, 'v0', 5, 40, 1).name('Desired speed (m/s)'),
      carFolder.add(car, 'a', 0.5, 4, 0.1).name('Max accel (m/s²)'),
      carFolder.add(car, 'b', 0.5, 6, 0.1).name('Comfy decel (m/s²)'),
      carFolder.add(car, 'T', 0.5, 3, 0.1).name('Time headway (s)'),
      carFolder.add(car, 's0', 0.5, 10, 0.5).name('Min distance (m)'),
      carFolder.add(car, 'delta', 1, 10, 1).name('Accel exponent'),
    );

    // Live speed/gap readout for the selected car, right under the sliders.
    Object.keys(state.telemetry.speeds).forEach((key) => {
      if (Number(key) >= state.settings.cars.length) delete state.telemetry.speeds[key];
    });
    state.telemetry.speeds[String(state.selectedCar)] ??= ''; // lil-gui needs the property to exist
    paramControllers.push(
      carFolder.add(state.telemetry.speeds, String(state.selectedCar)).name('Speed').listen(),
    );
  };

  const listController = carFolder
    .add(state, 'selectedCar', {})
    .name('Selected')
    .onChange(refreshCarPanel);
  const addController = carFolder
    .add(
      {
        addCar: () => {
          if (!state.canSpawn || state.settings.cars.length >= MAX_CARS) return;
          state.settings.cars.push({ ...NEW_CAR_PARAMS });
          state.selectedCar = state.settings.cars.length - 1;
          refreshCarPanel();
        },
      },
      'addCar',
    )
    .name('Add car');
  // Disabled when the renderer reports no safe spawn slot, or the car cap is reached.
  setInterval(
    () => addController.disable(!state.canSpawn || state.settings.cars.length >= MAX_CARS),
    250,
  );
  carFolder
    .add(
      {
        deleteCar: () => {
          if (state.settings.cars.length <= 1) return; // keep at least one car
          state.settings.cars.splice(state.selectedCar, 1);
          refreshCarPanel();
        },
      },
      'deleteCar',
    )
    .name('Delete car');

  gui.add(state.telemetry, 'light').name('Light').listen();
  refreshCarPanel();
  return state;
}
