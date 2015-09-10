#include <boost/test/unit_test.hpp>
#include <object_recognition/Dummy.hpp>

using namespace object_recognition;

BOOST_AUTO_TEST_CASE(it_should_not_crash_when_welcome_is_called)
{
    object_recognition::DummyClass dummy;
    dummy.welcome();
}
