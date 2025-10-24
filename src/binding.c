#include "territories.h"
#define MY_GET
#define MY_PUT
#define Env Territories
#include "../env_binding.h"

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
  env->n_genes = unpack(kwargs, "n_genes");
  env->width = unpack(kwargs, "width");
  env->height = unpack(kwargs, "height");
  env->max_agents = unpack(kwargs, "max_agents");
  env->n_roles = unpack(kwargs, "n_roles");
  env->min_ep_length = unpack(kwargs, "min_ep_length");
  env->max_ep_length = unpack(kwargs, "max_ep_length");
  env->render_mode = unpack(kwargs, "render_mode");
  env->extinction_reward = unpack(kwargs, "extinction_reward");
  env->n_alleles = unpack(kwargs, "n_alleles");
  env->reward_growth_rate = unpack(kwargs, "reward_growth_rate");

  // Extract optional map_name string
  PyObject *map_name_obj = PyDict_GetItemString(kwargs, "map_name");
  if (map_name_obj != NULL && map_name_obj != Py_None) {
    if (!PyUnicode_Check(map_name_obj)) {
      PyErr_SetString(PyExc_TypeError, "map_name must be a string");
      return -1;
    }
    const char *map_name_str = PyUnicode_AsUTF8(map_name_obj);
    if (map_name_str == NULL) {
      return -1;
    }
    strncpy(env->map_name, map_name_str, sizeof(env->map_name) - 1);
    env->map_name[sizeof(env->map_name) - 1] = '\0'; // Ensure null termination
  } else {
    // If not provided or None, set to empty string (will be treated as NULL by
    // read_is_soil)
    env->map_name[0] = '\0';
  }

  PyObject *alive = PyTuple_GetItem(args, 6);
  if (!PyObject_TypeCheck(alive, &PyArray_Type)) {
    PyErr_SetString(PyExc_TypeError, "Alive mask must be a NumPy array");
    return -1;
  }
  PyArrayObject *alive_mask = (PyArrayObject *)alive;
  if (!PyArray_ISCONTIGUOUS(alive_mask)) {
    PyErr_SetString(PyExc_ValueError, "Alive mask must be contiguous");
    return -1;
  }
  if (PyArray_NDIM(alive_mask) != 1) {
    PyErr_SetString(PyExc_ValueError, "Alive mask must be 1D");
    return -1;
  }
  env->alive_mask = PyArray_DATA(alive_mask);

  PyObject *kinship = PyTuple_GetItem(args, 7);
  if (!PyObject_TypeCheck(kinship, &PyArray_Type)) {
    PyErr_SetString(PyExc_TypeError, "DNAs must be a NumPy array");
    return -1;
  }
  PyArrayObject *kinship_matrix = (PyArrayObject *)kinship;
  if (!PyArray_ISCONTIGUOUS(kinship_matrix)) {
    PyErr_SetString(PyExc_ValueError, "DNAs must be contiguous");
    return -1;
  }
  env->kinship_matrix = PyArray_DATA(kinship_matrix);

  PyObject *dnas = PyTuple_GetItem(args, 8);
  if (!PyObject_TypeCheck(dnas, &PyArray_Type)) {
    PyErr_SetString(PyExc_TypeError, "DNAs must be a NumPy array");
    return -1;
  }
  PyArrayObject *dnas_array = (PyArrayObject *)dnas;
  if (!PyArray_ISCONTIGUOUS(dnas_array)) {
    PyErr_SetString(PyExc_ValueError, "DNAs must be contiguous");
    return -1;
  }
  if (PyArray_NDIM(dnas_array) != 2) {
    PyErr_SetString(PyExc_ValueError, "DNAs must be 2D");
    return -1;
  }
  if (PyArray_DIM(dnas_array, 0) != env->max_agents) {
    PyErr_SetString(PyExc_ValueError, "DNAs must have max_agents rows");
    return -1;
  }
  if (PyArray_DIM(dnas_array, 1) != env->n_genes) {
    PyErr_SetString(PyExc_ValueError, "DNAs must have n_genes columns");
    return -1;
  }
  env->dnas = PyArray_DATA(dnas_array);

  init(env);
  return 0;
}

static int my_log(PyObject *dict, Log *log) {
  assign_to_dict(dict, "births", log->births);
  assign_to_dict(dict, "starvations", log->starvations);
  assign_to_dict(dict, "murders", log->murders);
  assign_to_dict(dict, "stone_mined", log->stone_mined);
  assign_to_dict(dict, "walls_built", log->walls_built);
  assign_to_dict(dict, "wall_destroyed", log->wall_destroyed);
  assign_to_dict(dict, "food_stored", log->food_stored);
  assign_to_dict(dict, "food_eaten", log->food_eaten);
  assign_to_dict(dict, "max_pop", log->max_pop);
  assign_to_dict(dict, "min_pop", log->min_pop);
  assign_to_dict(dict, "avg_population", log->avg_population);
  assign_to_dict(dict, "total_reward", log->total_reward);
  assign_to_dict(dict, "episode_length", log->episode_length);
  assign_to_dict(dict, "life_expectancy", log->life_expectancy);
  assign_to_dict(dict, "genetic_diversity", log->genetic_diversity);
  return 0;
}

static PyObject *my_get(PyObject *dict, Env *env) {
  // Extract basic info
  PyDict_SetItemString(dict, "tick", PyLong_FromLong(env->tick));
  PyDict_SetItemString(dict, "is_winter", PyBool_FromLong(env->is_winter));

  // Extract episode stats
  PyDict_SetItemString(dict, "stats_births",
                       PyFloat_FromDouble(env->stats.births));
  PyDict_SetItemString(dict, "stats_starvations",
                       PyFloat_FromDouble(env->stats.starvations));
  PyDict_SetItemString(dict, "stats_murders",
                       PyFloat_FromDouble(env->stats.murders));
  PyDict_SetItemString(dict, "stats_stone_mined",
                       PyFloat_FromDouble(env->stats.stone_mined));
  PyDict_SetItemString(dict, "stats_walls_built",
                       PyFloat_FromDouble(env->stats.walls_built));
  PyDict_SetItemString(dict, "stats_wall_destroyed",
                       PyFloat_FromDouble(env->stats.wall_destroyed));
  PyDict_SetItemString(dict, "stats_food_stored",
                       PyFloat_FromDouble(env->stats.food_stored));
  PyDict_SetItemString(dict, "stats_food_eaten",
                       PyFloat_FromDouble(env->stats.food_eaten));
  PyDict_SetItemString(dict, "stats_max_pop",
                       PyFloat_FromDouble(env->stats.max_pop));
  PyDict_SetItemString(dict, "stats_min_pop",
                       PyFloat_FromDouble(env->stats.min_pop));
  PyDict_SetItemString(dict, "stats_avg_population",
                       PyFloat_FromDouble(env->stats.avg_population));

  // Extract tile_props as numpy array
  npy_intp tile_props_dims[] = {env->height, env->width, 4};
  PyObject *tile_props_array =
      PyArray_SimpleNew(3, tile_props_dims, NPY_UINT16);
  memcpy(PyArray_DATA((PyArrayObject *)tile_props_array), env->tile_props,
         env->height * env->width * 4 * sizeof(unsigned short));
  PyDict_SetItemString(dict, "tile_props", tile_props_array);
  Py_DECREF(tile_props_array);

  // Extract pids_2d as numpy array
  npy_intp pids_2d_dims[] = {env->height, env->width};
  PyObject *pids_2d_array = PyArray_SimpleNew(2, pids_2d_dims, NPY_INT16);
  memcpy(PyArray_DATA((PyArrayObject *)pids_2d_array), env->pids_2d,
         env->height * env->width * sizeof(short));
  PyDict_SetItemString(dict, "pids_2d", pids_2d_array);
  Py_DECREF(pids_2d_array);

  // Extract agent data as a structured array
  npy_intp agent_dims[] = {env->max_agents};

  // Agent positions
  PyObject *agent_r = PyArray_SimpleNew(1, agent_dims, NPY_INT32);
  PyObject *agent_c = PyArray_SimpleNew(1, agent_dims, NPY_INT32);
  PyObject *agent_dir = PyArray_SimpleNew(1, agent_dims, NPY_INT32);
  PyObject *agent_hp = PyArray_SimpleNew(1, agent_dims, NPY_INT32);
  PyObject *agent_hp_max = PyArray_SimpleNew(1, agent_dims, NPY_INT32);
  PyObject *agent_satiation = PyArray_SimpleNew(1, agent_dims, NPY_INT32);
  PyObject *agent_max_satiation = PyArray_SimpleNew(1, agent_dims, NPY_INT32);
  PyObject *agent_age = PyArray_SimpleNew(1, agent_dims, NPY_INT32);
  PyObject *agent_food = PyArray_SimpleNew(1, agent_dims, NPY_INT32);
  PyObject *agent_stone = PyArray_SimpleNew(1, agent_dims, NPY_INT32);
  PyObject *agent_role = PyArray_SimpleNew(1, agent_dims, NPY_INT32);

  int *r_data = (int *)PyArray_DATA((PyArrayObject *)agent_r);
  int *c_data = (int *)PyArray_DATA((PyArrayObject *)agent_c);
  int *dir_data = (int *)PyArray_DATA((PyArrayObject *)agent_dir);
  int *hp_data = (int *)PyArray_DATA((PyArrayObject *)agent_hp);
  int *hp_max_data = (int *)PyArray_DATA((PyArrayObject *)agent_hp_max);
  int *sat_data = (int *)PyArray_DATA((PyArrayObject *)agent_satiation);
  int *max_sat_data = (int *)PyArray_DATA((PyArrayObject *)agent_max_satiation);
  int *age_data = (int *)PyArray_DATA((PyArrayObject *)agent_age);
  int *food_data = (int *)PyArray_DATA((PyArrayObject *)agent_food);
  int *stone_data = (int *)PyArray_DATA((PyArrayObject *)agent_stone);
  int *role_data = (int *)PyArray_DATA((PyArrayObject *)agent_role);

  for (int i = 0; i < env->max_agents; i++) {
    r_data[i] = env->agents[i].r;
    c_data[i] = env->agents[i].c;
    dir_data[i] = env->agents[i].dir;
    hp_data[i] = env->agents[i].hp;
    hp_max_data[i] = env->agents[i].hp_max;
    sat_data[i] = env->agents[i].satiation;
    max_sat_data[i] = env->agents[i].max_satiation;
    age_data[i] = env->agents[i].age;
    food_data[i] = env->agents[i].food_carried;
    stone_data[i] = env->agents[i].stone_carried;
    role_data[i] = env->agents[i].role;
  }

  PyDict_SetItemString(dict, "agent_r", agent_r);
  PyDict_SetItemString(dict, "agent_c", agent_c);
  PyDict_SetItemString(dict, "agent_dir", agent_dir);
  PyDict_SetItemString(dict, "agent_hp", agent_hp);
  PyDict_SetItemString(dict, "agent_hp_max", agent_hp_max);
  PyDict_SetItemString(dict, "agent_satiation", agent_satiation);
  PyDict_SetItemString(dict, "agent_max_satiation", agent_max_satiation);
  PyDict_SetItemString(dict, "agent_age", agent_age);
  PyDict_SetItemString(dict, "agent_food_carried", agent_food);
  PyDict_SetItemString(dict, "agent_stone_carried", agent_stone);
  PyDict_SetItemString(dict, "agent_role", agent_role);

  Py_DECREF(agent_r);
  Py_DECREF(agent_c);
  Py_DECREF(agent_dir);
  Py_DECREF(agent_hp);
  Py_DECREF(agent_hp_max);
  Py_DECREF(agent_satiation);
  Py_DECREF(agent_max_satiation);
  Py_DECREF(agent_age);
  Py_DECREF(agent_food);
  Py_DECREF(agent_stone);
  Py_DECREF(agent_role);

  // Extract DNA data as numpy array
  npy_intp dna_dims[] = {env->max_agents, env->n_genes};
  PyObject *dna_array = PyArray_SimpleNew(2, dna_dims, NPY_UINT8);
  memcpy(PyArray_DATA((PyArrayObject *)dna_array), env->dnas,
         env->max_agents * env->n_genes * sizeof(unsigned char));
  PyDict_SetItemString(dict, "dnas", dna_array);
  Py_DECREF(dna_array);

  return NULL;
}

static int my_put(Env *env, PyObject *args, PyObject *kwargs) {
  // Inject state back into environment for replay
  if (!kwargs)
    return 0;

  PyObject *tick_obj = PyDict_GetItemString(kwargs, "tick");
  if (tick_obj)
    env->tick = PyLong_AsLong(tick_obj);

  PyObject *is_winter_obj = PyDict_GetItemString(kwargs, "is_winter");
  if (is_winter_obj)
    env->is_winter = PyObject_IsTrue(is_winter_obj);

  // Inject episode stats
  PyObject *stats_births_obj = PyDict_GetItemString(kwargs, "stats_births");
  if (stats_births_obj)
    env->stats.births = PyFloat_AsDouble(stats_births_obj);

  PyObject *stats_starvations_obj =
      PyDict_GetItemString(kwargs, "stats_starvations");
  if (stats_starvations_obj)
    env->stats.starvations = PyFloat_AsDouble(stats_starvations_obj);

  PyObject *stats_murders_obj = PyDict_GetItemString(kwargs, "stats_murders");
  if (stats_murders_obj)
    env->stats.murders = PyFloat_AsDouble(stats_murders_obj);

  PyObject *stats_stone_mined_obj =
      PyDict_GetItemString(kwargs, "stats_stone_mined");
  if (stats_stone_mined_obj)
    env->stats.stone_mined = PyFloat_AsDouble(stats_stone_mined_obj);

  PyObject *stats_walls_built_obj =
      PyDict_GetItemString(kwargs, "stats_walls_built");
  if (stats_walls_built_obj)
    env->stats.walls_built = PyFloat_AsDouble(stats_walls_built_obj);

  PyObject *stats_wall_destroyed_obj =
      PyDict_GetItemString(kwargs, "stats_wall_destroyed");
  if (stats_wall_destroyed_obj)
    env->stats.wall_destroyed = PyFloat_AsDouble(stats_wall_destroyed_obj);

  PyObject *stats_food_stored_obj =
      PyDict_GetItemString(kwargs, "stats_food_stored");
  if (stats_food_stored_obj)
    env->stats.food_stored = PyFloat_AsDouble(stats_food_stored_obj);

  PyObject *stats_food_eaten_obj =
      PyDict_GetItemString(kwargs, "stats_food_eaten");
  if (stats_food_eaten_obj)
    env->stats.food_eaten = PyFloat_AsDouble(stats_food_eaten_obj);

  PyObject *stats_pop_last_summer_day_obj =
      PyDict_GetItemString(kwargs, "stats_pop_last_summer_day");
  if (stats_pop_last_summer_day_obj)
    env->stats.max_pop = PyFloat_AsDouble(stats_pop_last_summer_day_obj);

  PyObject *stats_pop_last_winter_day_obj =
      PyDict_GetItemString(kwargs, "stats_pop_last_winter_day");
  if (stats_pop_last_winter_day_obj)
    env->stats.min_pop = PyFloat_AsDouble(stats_pop_last_winter_day_obj);

  PyObject *stats_avg_population_obj =
      PyDict_GetItemString(kwargs, "stats_avg_population");
  if (stats_avg_population_obj)
    env->stats.avg_population = PyFloat_AsDouble(stats_avg_population_obj);

  // Inject tile_props
  PyObject *tile_props_obj = PyDict_GetItemString(kwargs, "tile_props");
  if (tile_props_obj && PyArray_Check(tile_props_obj)) {
    PyArrayObject *arr = (PyArrayObject *)tile_props_obj;
    memcpy(env->tile_props, PyArray_DATA(arr),
           env->height * env->width * 4 * sizeof(unsigned short));
  }

  // Inject pids_2d
  PyObject *pids_2d_obj = PyDict_GetItemString(kwargs, "pids_2d");
  if (pids_2d_obj && PyArray_Check(pids_2d_obj)) {
    PyArrayObject *arr = (PyArrayObject *)pids_2d_obj;
    memcpy(env->pids_2d, PyArray_DATA(arr),
           env->height * env->width * sizeof(short));
  }

  // Inject agent data
  PyObject *agent_r = PyDict_GetItemString(kwargs, "agent_r");
  PyObject *agent_c = PyDict_GetItemString(kwargs, "agent_c");
  PyObject *agent_dir = PyDict_GetItemString(kwargs, "agent_dir");
  PyObject *agent_hp = PyDict_GetItemString(kwargs, "agent_hp");
  PyObject *agent_hp_max = PyDict_GetItemString(kwargs, "agent_hp_max");
  PyObject *agent_satiation = PyDict_GetItemString(kwargs, "agent_satiation");
  PyObject *agent_max_satiation =
      PyDict_GetItemString(kwargs, "agent_max_satiation");
  PyObject *agent_age = PyDict_GetItemString(kwargs, "agent_age");
  PyObject *agent_food = PyDict_GetItemString(kwargs, "agent_food_carried");
  PyObject *agent_stone = PyDict_GetItemString(kwargs, "agent_stone_carried");
  PyObject *agent_role = PyDict_GetItemString(kwargs, "agent_role");

  if (agent_r && PyArray_Check(agent_r)) {
    int *r_data = (int *)PyArray_DATA((PyArrayObject *)agent_r);
    int *c_data = (int *)PyArray_DATA((PyArrayObject *)agent_c);
    int *dir_data = (int *)PyArray_DATA((PyArrayObject *)agent_dir);
    int *hp_data = (int *)PyArray_DATA((PyArrayObject *)agent_hp);
    int *hp_max_data = (int *)PyArray_DATA((PyArrayObject *)agent_hp_max);
    int *sat_data = (int *)PyArray_DATA((PyArrayObject *)agent_satiation);
    int *max_sat_data =
        (int *)PyArray_DATA((PyArrayObject *)agent_max_satiation);
    int *age_data = (int *)PyArray_DATA((PyArrayObject *)agent_age);
    int *food_data = (int *)PyArray_DATA((PyArrayObject *)agent_food);
    int *stone_data = (int *)PyArray_DATA((PyArrayObject *)agent_stone);
    int *role_data = (int *)PyArray_DATA((PyArrayObject *)agent_role);

    for (int i = 0; i < env->max_agents; i++) {
      env->agents[i].r = r_data[i];
      env->agents[i].c = c_data[i];
      env->agents[i].dir = dir_data[i];
      env->agents[i].hp = hp_data[i];
      env->agents[i].hp_max = hp_max_data[i];
      env->agents[i].satiation = sat_data[i];
      env->agents[i].max_satiation = max_sat_data[i];
      env->agents[i].age = age_data[i];
      env->agents[i].food_carried = food_data[i];
      env->agents[i].stone_carried = stone_data[i];
      env->agents[i].role = role_data[i];
    }
  }

  // Inject DNA data
  PyObject *dnas_obj = PyDict_GetItemString(kwargs, "dnas");
  if (dnas_obj && PyArray_Check(dnas_obj)) {
    PyArrayObject *arr = (PyArrayObject *)dnas_obj;
    memcpy(env->dnas, PyArray_DATA(arr),
           env->max_agents * env->n_genes * sizeof(unsigned char));
  }

  return 0;
}