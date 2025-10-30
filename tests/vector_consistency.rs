use rust_gymnasium::{Env, Step, SyncVectorEnv, CartPoleEnv};

// Ensure a vector env with N=1 produces the same rollout as a single env
// when seeds and actions are the same.
#[test]
fn single_vs_vector_n1_same_rollout() {
    // Single env
    let mut single = CartPoleEnv::default();
    let (_obs_s, _info_s) = single.reset(Some(0));

    // Vector env with N=1
    let mut vec_env = SyncVectorEnv::new(1, || CartPoleEnv::default());
    let _obs_all = vec_env.reset_all(Some(0));

    // Use a fixed action sequence
    let actions = vec![1, 1, 0, 1, 0, 0, 1, 1, 1, 0];
    for a in actions {
        let s_single: Step<_> = single.step(a);
        let s_vec = vec_env.step_all(vec![a])[0].clone();
        assert_eq!(s_single.observation, s_vec.observation);
        assert!((s_single.reward - s_vec.reward).abs() < 1e-6);
        assert_eq!(s_single.terminated, s_vec.terminated);
        assert_eq!(s_single.truncated, s_vec.truncated);
        if s_single.terminated || s_single.truncated { break; }
    }
}

// Basic sanity for N=2 shape/length behavior
#[test]
fn vector_two_envs_steps_lengths() {
    let mut v = SyncVectorEnv::new(2, || CartPoleEnv::default());
    let obs_infos = v.reset_all(Some(123));
    assert_eq!(obs_infos.len(), 2);
    let steps = v.step_all(vec![0, 1]);
    assert_eq!(steps.len(), 2);
    // Ensure observations are 4-length arrays (CartPole obs)
    assert_eq!(steps[0].observation.len(), 4);
    assert_eq!(steps[1].observation.len(), 4);
}
