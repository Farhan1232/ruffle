//! Lifetime test for weak-keyed `flash.utils.Dictionary`.

use ruffle_test_framework::environment::Environment;
use ruffle_test_framework::options::TestOptions;
use ruffle_test_framework::runner::TestStatus;
use ruffle_test_framework::test::Test;
use ruffle_test_framework::vfs::{PhysicalFS, VfsPath};
use std::thread::sleep;

/// A `Dictionary` constructed with `weakKeys` must not keep its keys alive,
/// and an entry must not keep itself alive through its own value.
///
/// `test.swf` fills a weak dictionary with twenty entries whose keys are
/// referenced only by the dictionary and by the entry's own value - the
/// shape of a cache keyed on a display object that stores something reaching
/// that object - plus a key the movie keeps, a string key and an integer key.
/// A strong-keyed dictionary gets the same twenty entries as a control. Two
/// full collections are forced at frame 20; the movie reports at frame 30
/// and the trace is compared with `output.txt`: the unreferenced weak entries
/// are gone, the strong ones are not, and the kept key still resolves,
/// enumerates and deletes.
pub fn weak_dictionary_releases_unreferenced_keys(
    environment: &impl Environment,
) -> Result<(), libtest_mimic::Failed> {
    let test = &Test::from_options(
        TestOptions {
            num_frames: Some(40),
            output_path: "output.txt".into(),
            ..Default::default()
        },
        VfsPath::new(PhysicalFS::new(
            "tests/swfs/avm2/dictionary_weak_keys_release/",
        )),
        "dictionary_weak_keys_release".to_string(),
    )?;

    let mut runner = test.create_test_runner(environment)?;
    let mut frames = 0;
    let mut collected = false;
    loop {
        let status = runner.tick()?;
        if runner.is_preloaded() {
            frames += 1;
        }
        if frames == 20 && !collected {
            collected = true;
            let mut player = runner.player().lock().unwrap();
            // Two full cycles: the first decides which entries die, the
            // second sweeps what they held.
            player.collect_all_garbage();
            player.collect_all_garbage();
        }
        match status {
            TestStatus::Continue => {}
            TestStatus::Sleep(duration) => sleep(duration),
            TestStatus::Finished => break,
        }
    }

    if !collected {
        return Err("the movie never reached frame 20".into());
    }
    Ok(())
}
